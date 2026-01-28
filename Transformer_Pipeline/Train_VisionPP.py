"""
Train_VisionPP.py - Training script for VisionTS++ model on cyber threat data.

VisionTS++ is an extended version designed for multivariate forecasting,
using shared image space where all variables are rendered into a single image.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
VISIONTS_PATH = REPO_ROOT / "Transformers" / "Visual_Transformer"
sys.path.insert(0, str(VISIONTS_PATH))
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
import torch.nn as nn
from Cyber_Trend_Image_Dataset import CyberTrendImageDataset
from Cyber_Trend_VisionPP_Config import CyberVisionTSppConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from Models.VisionTSpp_Wrapper import create_visiontspp_model


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(all_preds: torch.Tensor, all_targets: torch.Tensor) -> tuple[float, float]:
    preds_flat = all_preds.view(-1)
    targets_flat = all_targets.view(-1)
    target_mean = torch.mean(targets_flat)

    # RSE: sqrt(sum((pred - true)^2)) / sqrt(sum((true - mean)^2))
    numerator_rse = torch.sqrt(torch.sum((preds_flat - targets_flat) ** 2))
    denominator_rse = torch.sqrt(torch.sum((targets_flat - target_mean) ** 2))
    rse = numerator_rse / (denominator_rse + 1e-7)

    # RAE: sum(|pred - true|) / sum(|true - mean|)
    numerator_rae = torch.sum(torch.abs(preds_flat - targets_flat))
    denominator_rae = torch.sum(torch.abs(targets_flat - target_mean))
    rae = numerator_rae / (denominator_rae + 1e-7)

    return rse.item(), rae.item()


def compute_additional_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    """Compute additional metrics for comprehensive evaluation."""
    mse = float(np.mean((preds - trues) ** 2))
    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(mse))

    # Correlation
    sigma_p = preds.std(axis=0)
    sigma_g = trues.std(axis=0)
    mean_p = preds.mean(axis=0)
    mean_g = trues.mean(axis=0)
    index = (sigma_g != 0) & (sigma_p != 0)
    if index.any():
        correlation = ((preds - mean_p) * (trues - mean_g)).mean(axis=0) / (sigma_p * sigma_g + 1e-7)
        corr = float(correlation[index].mean())
    else:
        corr = 0.0

    return {"mse": mse, "mae": mae, "rmse": rmse, "correlation": corr}


class TemporalAwareLoss(nn.Module):
    """Combined L1 + temporal difference + correlation loss for time series."""

    def __init__(self, temporal_weight: float = 0.3, corr_weight: float = 0.3, eps: float = 1e-7):
        super().__init__()
        self.temporal_weight = temporal_weight
        self.corr_weight = corr_weight
        self.eps = eps
        self.l1_loss = nn.L1Loss()

    def pearson_correlation_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Differentiable Pearson correlation loss computed per-feature.

        Args:
            pred: (batch, time, features)
            target: (batch, time, features)

        Returns:
            Loss = 1 - mean(per-feature correlation across time)
        """
        # Center the data along time dimension
        pred_mean = pred.mean(dim=1, keepdim=True)  # (B, 1, F)
        target_mean = target.mean(dim=1, keepdim=True)

        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        # Compute covariance and standard deviations
        covariance = (pred_centered * target_centered).sum(dim=1)  # (B, F)
        pred_std = torch.sqrt((pred_centered**2).sum(dim=1) + self.eps)
        target_std = torch.sqrt((target_centered**2).sum(dim=1) + self.eps)

        # Pearson correlation per feature: (B, F)
        correlation = covariance / (pred_std * target_std + self.eps)

        # Loss = 1 - mean correlation (want to maximize correlation)
        return 1.0 - correlation.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Static loss: absolute error
        static_loss = self.l1_loss(pred, target)

        # Temporal difference loss: match rate of change
        # Shape: (batch, time, features) -> differences along time axis
        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
        target_diff = target[:, 1:, :] - target[:, :-1, :]
        temporal_loss = self.l1_loss(pred_diff, target_diff)

        # Correlation loss: maximize per-feature correlation
        corr_loss = self.pearson_correlation_loss(pred, target)

        return static_loss + self.temporal_weight * temporal_loss + self.corr_weight * corr_loss


class VisionTSppTrainer:
    """Trainer for VisionTS++ on cyber threat data."""

    def __init__(self, config: CyberVisionTSppConfig, data_dir: Optional[str] = None) -> None:
        self.config = config
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")

        # Load data first to get dimensions
        self.train_loader, self.val_loader, self.test_loader = self._prepare_data()

        # Build model after we know the dimensions
        self.model = self._build_model()

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
        )
        self.criterion = TemporalAwareLoss(
            temporal_weight=config.temporal_weight,
            corr_weight=config.corr_weight,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)

    def _build_model(self):
        """Build the VisionTS++ model."""
        print(f"\n--- Building VisionTS++ Model ---")
        print(f"Architecture: {self.config.model_arch}")
        print(f"Fine-tune type: {self.config.finetune_type}")

        model_config = {
            "model_arch": self.config.model_arch,
            "finetune_type": self.config.finetune_type,
            "ckpt_dir": str(self.config.ckpt_dir),
            "load_pretrained": self.config.load_pretrained,
            "periodicity": self.config.periodicity,
            "norm_const": self.config.norm_const,
            "align_const": self.config.align_const,
            "interpolation": self.config.interpolation,
            "padding_mode": self.config.padding_mode,
            # VisionTS++ specific
            "quantile": self.config.quantile,
            "quantile_head_num": self.config.quantile_head_num,
            "color": self.config.color,
            "clip_input": self.config.clip_input,
            "complete_no_clip": self.config.complete_no_clip,
            "num_patch_input": self.config.num_patch_input,
        }
        model = create_visiontspp_model(
            config=model_config,
            context_len=self.context_len,
            pred_len=self.pred_len,
            num_features=self.num_features,
            device=str(self.device),
        )
        return model

    def _prepare_data(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        if self.data_dir is None:
            raise ValueError("Data directory must be specified and valid.")

        print(f"\n--- Loading Preprocessed Data ---")
        print(f"Data directory: {self.data_dir}")
        train_dataset = CyberTrendImageDataset(data_dir=self.data_dir, split="train")
        val_dataset = CyberTrendImageDataset(data_dir=self.data_dir, split="val")
        test_dataset = CyberTrendImageDataset(data_dir=self.data_dir, split="test")

        # Get dimensions from dataset
        sample = train_dataset[0]
        self.context_len = sample["input"].shape[0]
        self.pred_len = sample["target"].shape[0]
        self.num_features = sample["input"].shape[1]

        print(f"Dimensions: context_len={self.context_len}, pred_len={self.pred_len}, nvars={self.num_features}")
        print(f"Samples: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

        # Validate feature count for VisionTS++
        if self.num_features > 224:
            print(f"WARNING: {self.num_features} features exceeds VisionTS++ limit of 224!")

        # Validate datasets
        for name, ds in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
            if len(ds) == 0:
                raise ValueError(f"{name} dataset is empty!")

        # Create data loaders
        drop_last_train = len(train_dataset) >= 2 * self.config.batch_size

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=drop_last_train,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        print(f"Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
        return train_loader, val_loader, test_loader

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}", leave=False)
        for batch in pbar:
            x = batch["input"].to(self.device)
            y_true = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y_true)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> tuple[float, float, float]:
        """Validate and return loss, RSE, RAE."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            x = batch["input"].to(self.device)
            y_true = batch["target"].to(self.device)

            y_pred = self.model(x)
            loss = self.criterion(y_pred, y_true)

            total_loss += loss.item()
            all_preds.append(y_pred.cpu())
            all_targets.append(y_true.cpu())

        avg_loss = total_loss / len(self.val_loader)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        rse, rae = compute_metrics(all_preds, all_targets)

        return avg_loss, rse, rae

    @torch.no_grad()
    def test(self) -> dict[str, float]:
        """Evaluate on test set and return metrics."""
        self.model.eval()
        all_preds = []
        all_targets = []

        for batch in tqdm(self.test_loader, desc="Testing"):
            x = batch["input"].to(self.device)
            y_true = batch["target"].to(self.device)

            y_pred = self.model(x)

            all_preds.append(y_pred.cpu())
            all_targets.append(y_true.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute primary metrics
        rse, rae = compute_metrics(all_preds, all_targets)

        # Compute additional metrics
        preds_np = all_preds.numpy()
        trues_np = all_targets.numpy()
        additional = compute_additional_metrics(preds_np, trues_np)

        metrics = {
            "rse": rse,
            "rae": rae,
            **additional,
        }

        # Print results
        print(f"\n{'=' * 50}")
        print("Test Results")
        print(f"{'=' * 50}")
        print(f"RSE:         {rse:.6f}")
        print(f"RAE:         {rae:.6f}")
        print(f"MSE:         {additional['mse']:.6f}")
        print(f"MAE:         {additional['mae']:.6f}")
        print(f"RMSE:        {additional['rmse']:.6f}")
        print(f"Correlation: {additional['correlation']:.6f}")

        # Save results
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "predictions.npy", preds_np)
        np.save(output_dir / "ground_truth.npy", trues_np)
        print(f"\nResults saved to {output_dir}")

        return metrics

    def train(self) -> None:
        """Run full training loop."""
        print(f"\n{'=' * 50}")
        print("Starting VisionTS++ Training")
        print(f"{'=' * 50}")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Training
            train_loss = self.train_epoch(epoch)

            # Validation
            val_loss, val_rse, val_rae = self.validate()

            self.scheduler.step()
            epoch_time = time.time() - start_time

            print(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"RSE: {val_rse:.4f} | RAE: {val_rae:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                checkpoint_path = Path(self.config.checkpoint_dir) / "visiontspp_best.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_rse": val_rse,
                        "val_rae": val_rae,
                    },
                    checkpoint_path,
                )
                print(f"Saved best model.")
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print(f"\n{'=' * 50}")
        print("Training Complete")
        print(f"{'=' * 50}")

    def zero_shot_eval(self) -> dict[str, float]:
        """Run zero-shot evaluation (no training)."""
        print(f"\n{'=' * 50}")
        print("Zero-Shot Evaluation")
        print(f"{'=' * 50}")
        return self.test()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train/test VisionTS++ model for cyber threat forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data_dir", type=str, default=None, help="Path to preprocessed data directory")

    # Training parameters
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--patience", type=int, help="Early stopping patience")

    # Mode and architecture
    parser.add_argument("--mode", type=str, choices=["train", "test", "zero_shot"], help="Execution mode")
    parser.add_argument(
        "--model_arch", type=str, choices=["mae_base", "mae_large", "mae_huge"], help="Model architecture"
    )
    parser.add_argument(
        "--finetune_type", type=str, choices=["none", "ln", "bias", "full", "mlp", "attn"], help="Fine-tuning strategy"
    )

    # VisionTS++ specific
    parser.add_argument("--quantile", action="store_true", help="Enable quantile predictions")
    parser.add_argument("--no_color", action="store_true", help="Disable color encoding")
    parser.add_argument("--norm_const", type=float, help="Normalization constant")
    parser.add_argument("--align_const", type=float, help="Alignment constant")

    # System parameters
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, help="Number of data loader workers")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = CyberVisionTSppConfig()
    config.update_from_dict(vars(args))

    # Handle special args
    if args.no_color:
        config.color = False
    if args.quantile:
        config.quantile = True

    # Set random seed
    set_random_seed(config.seed)
    print(f"Random seed: {config.seed}")

    # Resolve data directory
    data_dir = args.data_dir
    if data_dir is None:
        # Default to visionpp processed data
        data_dir = str(config.output_dir)
    elif not Path(data_dir).is_absolute():
        data_dir = str(SCRIPT_DIR / data_dir)

    # Create trainer
    trainer = VisionTSppTrainer(config, data_dir=data_dir)

    # Run based on mode
    if config.mode == "zero_shot":
        trainer.zero_shot_eval()
    elif config.mode == "train":
        trainer.train()
        # Load best model and test
        checkpoint_path = Path(config.checkpoint_dir) / "visiontspp_best.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
        trainer.test()
    elif config.mode == "test":
        checkpoint_path = Path(config.checkpoint_dir) / "visiontspp_best.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from {checkpoint_path}")
        trainer.test()


if __name__ == "__main__":
    main()
