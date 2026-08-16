from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn

# Add project root to python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
VISIONTS_PATH = os.path.join(PROJECT_ROOT, "Transformers", "Visual_Transformer")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if VISIONTS_PATH not in sys.path:
    sys.path.insert(0, VISIONTS_PATH)

try:
    from visionts import VisionTSpp
except ImportError as e:
    print(f"Error: Could not import VisionTSpp from submodule.")
    print(f"Please ensure the submodule is initialised: `git submodule update --init --recursive`")
    print(f"Expected path: {VISIONTS_PATH}")
    raise e


class VisionTSppModel(nn.Module):
    """
    A PyTorch nn.Module wrapper for VisionTS++ (extended multivariate model).

    Key differences from standard VisionTS:
      - Designed for multivariate forecasting with shared image space
      - All variables rendered into a single image (max 224 features due to image size)
      - Uses color channels to encode different variables
      - Supports quantile predictions for uncertainty estimation
      - Downloads pretrained checkpoint from HuggingFace

    Input shape: (batch, context_len, nvars) - 3D tensors
    Output shape: (batch, pred_len, nvars) - 3D tensors
                  or [(batch, pred_len, nvars), [quantiles...]] if quantile=True
    """

    def __init__(
        self,
        config: dict,
        context_len: int,
        pred_len: int,
        num_features: int,
    ):
        super().__init__()

        self.config = config
        self.context_len = context_len
        self.pred_len = pred_len
        self.num_features = num_features

        # Extract VisionTS++ specific config
        model_arch = config.get("model_arch", "mae_base")
        finetune_type = config.get("finetune_type", "ln")
        ckpt_dir = config.get("ckpt_dir", "./ckpt/")
        load_pretrained = config.get("load_pretrained", True)

        # VisionTS++ specific options
        self.quantile = config.get("quantile", False)
        self.quantile_head_num = config.get("quantile_head_num", 9)
        self.color = config.get("color", True)
        self.clip_input = config.get("clip_input", 0)
        self.complete_no_clip = config.get("complete_no_clip", False)

        # Chunked rendering: render at most chunk_size variables per image and concatenate the per-chunk forecasts.
        # With N features each variable gets int(224/N) image rows, so at 159 features every series is a single row.
        # None = render all features in one image (original behaviour).
        self.chunk_size = config.get("chunk_size", None)

        print(f"Init VisionTS++ wrapper:")
        print(f"  - Architecture: {model_arch}")
        print(f"  - Fine-tune type: {finetune_type}")
        print(f"  - Context length: {context_len}")
        print(f"  - Prediction length: {pred_len}")
        print(f"  - Number of features: {num_features}")
        print(f"  - Quantile prediction: {self.quantile}")
        print(f"  - Color encoding: {self.color}")

        # Validate feature count (VisionTS++ uses shared image space)
        if num_features > 224:
            print(f"  WARNING: {num_features} features exceeds VisionTS++ limit of 224")
            print(f"           Consider using feature selection to reduce dimensions")

        # Ensure checkpoint directory exists
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

        self.visionts_engine = VisionTSpp(
            arch=model_arch,
            finetune_type=finetune_type,
            ckpt_dir=ckpt_dir,
            load_ckpt=load_pretrained,
            quantile=self.quantile,
            quantile_head_num=self.quantile_head_num,
            color=self.color,
            clip_input=self.clip_input,
            complete_no_clip=self.complete_no_clip,
        )

        # Configure for our specific task
        periodicity = config.get("periodicity", 1)
        norm_const = config.get("norm_const", 0.4)
        align_const = config.get("align_const", 0.4)
        interpolation = config.get("interpolation", "bilinear")
        num_patch_input = config.get("num_patch_input", None)
        padding_mode = config.get("padding_mode", "replicate")

        self.visionts_engine.update_config(
            context_len=context_len,
            pred_len=pred_len,
            num_patch_input=num_patch_input,
            periodicity=periodicity,
            norm_const=norm_const,
            align_const=align_const,
            interpolation=interpolation,
            padding_mode=padding_mode,
        )

        print(f"  - Periodicity: {periodicity}")
        print(f"  - Norm const: {norm_const}")
        print(f"  - Align const: {align_const}")
        print(f"  - Chunk size: {self.chunk_size if self.chunk_size else 'off (single image)'}")

        # Count parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"VisionTS++ initialized. Parameters: {trainable_params:,} trainable / {total_params:,} total")

    def _feature_chunks(self, nvars: int):
        """Contiguous feature slices of at most self.chunk_size variables."""
        cs = self.chunk_size
        return [(i, min(i + cs, nvars)) for i in range(0, nvars, cs)]

    def _engine_forward(self, x: torch.Tensor):
        """Run the engine, chunking over features when chunk_size is set.

        Returns a tensor, or [main_pred, [quantile_preds...]] in quantile mode.
        """
        if not self.chunk_size or x.shape[-1] <= self.chunk_size:
            return self.visionts_engine(x)

        mains, quant_chunks = [], []
        for lo, hi in self._feature_chunks(x.shape[-1]):
            out = self.visionts_engine(x[:, :, lo:hi])
            if self.quantile and isinstance(out, list):
                mains.append(out[0])
                quant_chunks.append(out[1])
            else:
                mains.append(out)
        main = torch.cat(mains, dim=-1)
        if not quant_chunks:
            return main
        quantiles = [torch.cat(qs, dim=-1) for qs in zip(*quant_chunks)]
        return [main, quantiles]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VisionTS++.

        Args:
            x: Input tensor of shape (batch, context_len, nvars)

        Returns:
            Predictions tensor of shape (batch, pred_len, nvars)
            If quantile=True, returns [predictions, [quantile_predictions...]]
        """
        y_pred = self._engine_forward(x)

        # If quantile mode, extract just the main prediction for standard usage
        if self.quantile and isinstance(y_pred, list):
            # y_pred = [main_pred, [quantile_preds...]]
            return y_pred[0]

        return y_pred

    def forward_with_quantiles(self, x: torch.Tensor) -> tuple:
        """
        Forward pass returning both predictions and quantiles.

        Args:
            x: Input tensor of shape (batch, context_len, nvars)

        Returns:
            tuple: (predictions, quantile_list) if quantile=True else just predictions
        """
        if not self.quantile:
            return self._engine_forward(x), None

        result = self._engine_forward(x)
        if isinstance(result, list):
            return result[0], result[1]
        return result, None

    def forward_with_images(self, x: torch.Tensor) -> tuple:
        """
        Forward pass with image exports for visualization.

        Args:
            x: Input tensor of shape (batch, context_len, nvars)

        Returns:
            tuple: (predictions, input_image, reconstructed_image, nvars, color_list)
        """
        if self.chunk_size and x.shape[-1] > self.chunk_size:
            raise NotImplementedError(
                "forward_with_images does not support chunked rendering; "
                "call with chunk_size=None or <=chunk_size features")
        return self.visionts_engine(x, export_image=True)

    def update_config(self, **kwargs):
        """
        Update VisionTS++ configuration parameters.

        Useful for changing context_len or pred_len at runtime.
        """
        if "context_len" in kwargs:
            self.context_len = kwargs["context_len"]
        if "pred_len" in kwargs:
            self.pred_len = kwargs["pred_len"]

        self.visionts_engine.update_config(**kwargs)


def create_visiontspp_model(
    config: dict,
    context_len: int = 10,
    pred_len: int = 36,
    num_features: int = 25,
    device: str = "cpu",
) -> VisionTSppModel:
    """
    Factory function to create a VisionTS++ model.

    Args:
        config: Model configuration dictionary
        context_len: Input sequence length
        pred_len: Prediction horizon
        num_features: Number of input features (max 224 for VisionTS++)
        device: Device to place model on

    Returns:
        Initialized VisionTSppModel on specified device
    """
    model = VisionTSppModel(
        config=config,
        context_len=context_len,
        pred_len=pred_len,
        num_features=num_features,
    )
    return model.to(device)


# ------------------- Example Usage -------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VisionTS++ Wrapper - Example Usage")
    print("=" * 60)

    # Example configuration
    config = {
        "model_arch": "mae_base",
        "finetune_type": "ln",
        "ckpt_dir": os.path.join(PROJECT_ROOT, "Processed_Data", "visionpp", "ckpt"),
        "load_pretrained": True,
        "periodicity": 1,
        "norm_const": 0.4,
        "align_const": 0.4,
        "interpolation": "bilinear",
        "quantile": False,
        "color": True,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    try:
        model = create_visiontspp_model(
            config=config,
            context_len=10,
            pred_len=36,
            num_features=25,
            device=device,
        )

        print("\n--- Testing Forward Pass ---")
        batch_size = 4
        context_len = 10
        num_features = 25

        dummy_input = torch.randn(batch_size, context_len, num_features).to(device)
        print(f"Input shape: {dummy_input.shape}")

        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: ({batch_size}, 36, {num_features})")

        print("\nVisionTS++ wrapper test successful!")

    except Exception as e:
        print(f"\nError during test: {e}")
        print("This is expected if the VisionTS++ checkpoint hasn't been downloaded yet.")
        print("The checkpoint will be downloaded automatically during training.")
