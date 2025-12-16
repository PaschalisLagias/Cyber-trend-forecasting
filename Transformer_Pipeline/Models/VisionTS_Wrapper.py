from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Add project root to python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
VISIONTS_PATH = os.path.join(PROJECT_ROOT, "Transformers", "Visual_Transformer")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if VISIONTS_PATH not in sys.path:
    sys.path.insert(0, VISIONTS_PATH)

try:
    from visionts import VisionTS
except ImportError as e:
    print(f"Error: Could not import VisionTS from submodule.")
    print(f"Please ensure the submodule is initialised: `git submodule update --init --recursive`")
    print(f"Expected path: {VISIONTS_PATH}")
    raise e


class VisionTSModel(nn.Module):
    """
    A PyTorch nn.Module wrapper for the VisionTS model following the PDFormer interface.

    Primary differences from PDFormer wrapper:
      - VisionTS handles normalization internally (no external scaler needed)
      - Input shape: (batch, context_len, nvars) - 3D tensors
      - Output shape: (batch, pred_len, nvars) - 3D tensors
      - Must call update_config() before forward pass

    PREDICTION FLOW:
      1. Input tensor is passed to VisionTS
      2. VisionTS internally normalises (instance norm)
      3. Converts to 2D image representation
      4. Processes through MAE (Masked Autoencoder)
      5. Reconstructs and denormalisess output
      6. Returns predictions in original scale
    """

    def __init__(
        self,
        config: dict,
        context_len: int,
        pred_len: int,
        num_features: Optional[int] = None,
    ):
        super().__init__()

        self.config = config
        self.context_len = context_len
        self.pred_len = pred_len
        self.num_features = num_features

        # Extract VisionTS-specific config
        model_arch = config.get("model_arch", "mae_base")
        finetune_type = config.get("finetune_type", "ln")
        ckpt_dir = config.get("ckpt_dir", "./ckpt/")
        load_pretrained = config.get("load_pretrained", True)

        print(f"Init VisionTS wrapper:")
        print(f"  - Architecture: {model_arch}")
        print(f"  - Fine-tune type: {finetune_type}")
        print(f"  - Context length: {context_len}")
        print(f"  - Prediction length: {pred_len}")
        if num_features:
            print(f"  - Number of features: {num_features}")

        # Ensure checkpoint directory exists
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

        # Initialize the VisionTS model
        self.visionts_engine = VisionTS(
            arch=model_arch,
            finetune_type=finetune_type,
            ckpt_dir=ckpt_dir,
            load_ckpt=load_pretrained,
        )

        # Configure for our specific task
        periodicity = config.get("periodicity", 1)
        norm_const = config.get("norm_const", 0.4)
        align_const = config.get("align_const", 0.4)
        interpolation = config.get("interpolation", "bilinear")

        self.visionts_engine.update_config(
            context_len=context_len,
            pred_len=pred_len,
            periodicity=periodicity,
            norm_const=norm_const,
            align_const=align_const,
            interpolation=interpolation,
        )

        print(f"  - Periodicity: {periodicity}")
        print(f"  - Norm const: {norm_const}")
        print(f"  - Align const: {align_const}")

        # Count parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"VisionTS initialized. Parameters: {trainable_params:,} trainable / {total_params:,} total")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VisionTS.

        Args:
            x: Input tensor of shape (batch, context_len, nvars)

        Returns:
            Predictions tensor of shape (batch, pred_len, nvars)
        """
        # VisionTS expects (batch, context_len, nvars) and returns (batch, pred_len, nvars)
        y_pred = self.visionts_engine(x)
        return y_pred

    def forward_with_images(self, x: torch.Tensor) -> tuple:
        """
        Forward pass with image returns for visualization.

        Args:
            x: Input tensor of shape (batch, context_len, nvars)

        Returns:
            tuple: (predictions, input_image, reconstructed_image)
        """
        return self.visionts_engine(x, export_image=True)

    def update_config(self, **kwargs):
        """
        Update VisionTS configuration parameters.

        Useful for changing context_len or pred_len at runtime.
        """
        # Update stored values
        if "context_len" in kwargs:
            self.context_len = kwargs["context_len"]
        if "pred_len" in kwargs:
            self.pred_len = kwargs["pred_len"]

        self.visionts_engine.update_config(**kwargs)


def create_visionts_model(
    config: dict,
    context_len: int = 10,
    pred_len: int = 36,
    num_features: Optional[int] = None,
    device: str = "cpu",
) -> VisionTSModel:
    """
    Factory function to create a VisionTS model.

    Args:
        config: Model configuration dictionary
        context_len: Input sequence length
        pred_len: Prediction horizon
        num_features: Number of input features
        device: Device to place model on

    Returns:
        Initialized VisionTSModel on specified device
    """
    model = VisionTSModel(
        config=config,
        context_len=context_len,
        pred_len=pred_len,
        num_features=num_features,
    )
    return model.to(device)


# ------------------- Example Usage -------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VisionTSModel Wrapper - Example Usage")
    print("=" * 60)

    # Example configuration
    config = {
        "model_arch": "mae_base",
        "finetune_type": "ln",
        "ckpt_dir": os.path.join(PROJECT_ROOT, "Processed_Data", "vision", "ckpt"),
        "load_pretrained": True,
        "periodicity": 1,
        "norm_const": 0.4,
        "align_const": 0.4,
        "interpolation": "bilinear",
    }

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    try:
        model = create_visionts_model(
            config=config,
            context_len=10,
            pred_len=36,
            num_features=163,  # Cyber threat features
            device=device,
        )

        # Generate dummy input and test forward pass
        print("\n--- Testing Forward Pass ---")
        batch_size = 4
        context_len = 10
        num_features = 163

        dummy_input = torch.randn(batch_size, context_len, num_features).to(device)
        print(f"Input shape: {dummy_input.shape}")

        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: ({batch_size}, 36, {num_features})")

        print("\nVisionTS wrapper test successful!")

    except Exception as e:
        print(f"\nError during test: {e}")
        print("This is expected if the VisionTS checkpoint hasn't been downloaded yet.")
        print("The checkpoint will be downloaded automatically during training.")
