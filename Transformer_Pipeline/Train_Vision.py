#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
VISIONTS_PATH = REPO_ROOT / "Transformers" / "Visual_Transformer"
sys.path.insert(0, str(VISIONTS_PATH))
sys.path.insert(0, str(SCRIPT_DIR))


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Cyber_Trend_Image_Dataset import CyberTrendImageDataset
from Cyber_Trend_Vision_Config import CyberVisionTSConfig
from long_term_tsf.utils.metrics import MAE, MSE, RMSE, RSE
from Preprocessing.Cyber_Trend_to_Image import split_data
from Preprocessing.Load_Data import load_cyber_threat_data
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from visionts import VisionTS


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def CORR(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute correlation metric with zero-variance handling."""
    sigma_p = pred.std(axis=0)
    sigma_g = true.std(axis=0)
    mean_p = pred.mean(axis=0)
    mean_g = true.mean(axis=0)
    index = (sigma_g != 0) & (sigma_p != 0)
    correlation = ((pred - mean_p) * (true - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    return correlation[index].mean()


def compute_rae(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Relative Absolute Error."""
    pred = torch.tensor(predictions.flatten())
    true = torch.tensor(ground_truth.flatten())

    sum_absolute_diff = torch.sum(torch.abs(true - pred))
    mean_true = torch.mean(true)
    sum_absolute_r = torch.sum(torch.abs(true - mean_true))

    if sum_absolute_r == 0:
        return float("inf")

    rae = sum_absolute_diff / sum_absolute_r
    return rae.item()


def compute_smape(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Symmetric Mean Absolute Percentage Error."""
    smape = 0
    n_samples, pred_len, n_vars = ground_truth.shape

    for x in range(n_samples):
        for z in range(n_vars):
            numerator = np.abs(ground_truth[x, :, z] - predictions[x, :, z])
            denominator = (np.abs(ground_truth[x, :, z]) + np.abs(predictions[x, :, z])) / 2
            mask = denominator != 0
            if mask.any():
                smape += (numerator[mask] / denominator[mask]).mean()

    smape /= n_samples * n_vars
    return float(smape)


class VisionTSTrainer:
    """Trainer for VisionTS on cyber threat data."""

    def __init__(self, config: CyberVisionTSConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")

        self.model = self._build_model()

        self.train_loader, self.val_loader, self.test_loader = self._prepare_data()

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.learning_rate)
        self.criterion = nn.MSELoss()

    def _build_model(self) -> VisionTS:
        print(f"Building VisionTS model with architecture: {self.config.model_arch}")
        model = VisionTS(
            arch=self.config.model_arch,
            finetune_type=self.config.finetune_type,
            ckpt_dir=self.config.ckpt_dir,
            load_ckpt=self.config.load_pretrained,
        )

        model.update_config(
            context_len=self.config.context_len,
            pred_len=self.config.pred_len,
            periodicity=self.config.periodicity,
            norm_const=self.config.norm_const,
            align_const=self.config.align_const,
            interpolation=self.config.interpolation,
        )

        model = model.to(self.device)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

        return model

    def _prepare_data(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        print("Loading cyber threat data...")
        df_raw = load_cyber_threat_data(str(self.config.raw_data_path))
        if df_raw is None:
            raise ValueError("Failed to load data")
        print(f"Data shape: {df_raw.shape}")
        print(f"Date range: {df_raw.index.min()} to {df_raw.index.max()}")

        train_df, val_df, test_df = split_data(df_raw, self.config.train_split, self.config.val_split)
        print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        train_dataset = CyberTrendImageDataset(
            data=train_df,
            context_window_size=self.config.context_window_size,
            pred_window_size=self.config.pred_window_size,
            is_train=True,
        )

        val_dataset = CyberTrendImageDataset(
            data=val_df,
            context_window_size=self.config.context_window_size,
            pred_window_size=self.config.pred_window_size,
            is_train=False,
        )

        test_dataset = CyberTrendImageDataset(
            data=test_df,
            context_window_size=self.config.context_window_size,
            pred_window_size=self.config.pred_window_size,
            is_train=False,
        )

        print(f"Dataset samples: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        if len(train_dataset) == 0:
            raise ValueError(
                f"Training dataset is empty! "
                f"Check data split ({self.config.train_split}) and window sizes "
                f"(context={self.config.context_window_size}, pred={self.config.pred_window_size})"
            )
        if len(val_dataset) == 0:
            raise ValueError("Validation dataset is empty! Check your data split and window sizes.")
        if len(test_dataset) == 0:
            raise ValueError("Test dataset is empty! Check your data split and window sizes.")

        drop_last_train = len(train_dataset) >= 2 * self.config.batch_size

        if not drop_last_train:
            print(
                f"Warning: Small training dataset ({len(train_dataset)} samples, "
                f"batch_size={self.config.batch_size}). Setting drop_last=False for training."
            )

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
            drop_last=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            drop_last=False,
        )

        print(f"DataLoader batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

        if len(train_loader) == 0:
            raise ValueError(
                f"Training dataloader is empty! Dataset has {len(train_dataset)} samples, "
                f"batch_size={self.config.batch_size}. Try reducing batch_size or increasing data."
            )

        return train_loader, val_loader, test_loader

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
        for batch_idx, batch in enumerate(pbar):
            x = batch["input"].to(self.device)
            y_true = batch["target"].to(self.device)

            y_pred = self.model(x)
            loss = self.criterion(y_pred, y_true)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        for batch in tqdm(self.val_loader, desc="Validating"):
            x = batch["input"].to(self.device)
            y_true = batch["target"].to(self.device)

            y_pred = self.model(x)
            loss = self.criterion(y_pred, y_true)

            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    @torch.no_grad()
    def test(self) -> dict[str, float]:
        self.model.eval()

        all_preds = []
        all_trues = []

        for batch in tqdm(self.test_loader, desc="Testing"):
            x = batch["input"].to(self.device)
            y_true = batch["target"].to(self.device)

            y_pred = self.model(x)

            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y_true.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)

        mse = float(MSE(preds, trues))
        mae = float(MAE(preds, trues))
        rmse = float(RMSE(preds, trues))
        rse = float(RSE(preds.flatten(), trues.flatten()))
        rae = compute_rae(preds, trues)
        corr = float(np.mean(CORR(preds, trues)))
        smape = compute_smape(preds, trues)

        print(f"\nTest Results:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"RSE: {rse:.6f}")
        print(f"RAE: {rae:.6f}")
        print(f"Correlation: {corr:.6f}")
        print(f"sMAPE: {smape:.6f}")

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "predictions.npy", preds)
        np.save(output_dir / "ground_truth.npy", trues)
        np.save(output_dir / "metrics.npy", np.array([mse, mae, rmse, rse, rae, corr, smape]))

        print(f"\nResults saved to {output_dir}")

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "rse": rse,
            "rae": rae,
            "correlation": corr,
            "smape": smape,
        }

    def train(self) -> None:
        print(f"\n{'=' * 50}\nStarting Training\n{'=' * 50}")

        best_val_loss = float("inf")
        patience_counter = 0
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch + 1}/{self.config.epochs} - Train Loss: {train_loss:.6f}")

            val_loss = self.validate()
            print(f"Epoch {epoch + 1}/{self.config.epochs} - Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                checkpoint_path = Path(self.config.checkpoint_dir) / "best_model.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    checkpoint_path,
                )
                print(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        print(f"\n{'=' * 50}\nTraining Complete\n{'=' * 50}")

    def zero_shot_eval(self) -> dict[str, float]:
        print(f"\n{'=' * 50}\nStarting Zero-Shot Evaluation\n{'=' * 50}")
        return self.test()


def run_single_experiment(config: CyberVisionTSConfig) -> dict[str, float]:
    trainer = VisionTSTrainer(config)

    if config.mode == "zero_shot":
        metrics = trainer.zero_shot_eval()
    elif config.mode == "train":
        print("WARN: Untested with latest code changes")
        trainer.train()
        metrics = trainer.test()
    elif config.mode == "test":
        print("WARN: Untested with latest code changes")
        checkpoint_path = Path(config.checkpoint_dir) / "best_model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}")
        metrics = trainer.test()

    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments to override config defaults."""
    parser = argparse.ArgumentParser(
        description="Train/test VisionTS model for cyber threat forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--patience", type=int, help="Early stopping patience")

    # Mode and architecture
    parser.add_argument("--mode", type=str, choices=["train", "test", "zero_shot"],
                       help="Execution mode")
    parser.add_argument("--model_arch", type=str, choices=["mae_base", "mae_large", "mae_huge"],
                       help="Model architecture")
    parser.add_argument("--finetune_type", type=str,
                       choices=["none", "ln", "bias", "full", "mlp", "attn"],
                       help="Fine-tuning strategy")

    # System parameters
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, help="Number of data loader workers")
    parser.add_argument("--use_gpu", type=lambda x: x.lower() == "true",
                       help="Use GPU if available (true/false)")

    # Boolean flags
    parser.add_argument("--load_pretrained", action="store_true", default=None,
                       help="Load pretrained weights")
    parser.add_argument("--no_load_pretrained", dest="load_pretrained", action="store_false",
                       help="Do not load pretrained weights")

    # Data split ratios
    parser.add_argument("--train_split", type=float, help="Training data split ratio")
    parser.add_argument("--val_split", type=float, help="Validation data split ratio")
    parser.add_argument("--test_split", type=float, help="Test data split ratio")

    # Model hyperparameters
    parser.add_argument("--periodicity", type=int, help="Periodicity for seasonality")
    parser.add_argument("--norm_const", type=float, help="Normalization constant")
    parser.add_argument("--align_const", type=float, help="Alignment constant")
    parser.add_argument("--interpolation", type=str, help="Interpolation method")

    # Dataset parameters
    parser.add_argument("--context_window_size", type=int, help="Context window size")
    parser.add_argument("--pred_window_size", type=int, help="Prediction window size")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = CyberVisionTSConfig()

    config.update_from_dict(vars(args))

    set_random_seed(config.seed)
    print(f"Random seed set to: {config.seed}")

    run_single_experiment(config)


if __name__ == "__main__":
    main()

### UNUSED METHODS BELOW THIS LINE - KEPT FOR REFERENCE ###
# def normalize_data(
#     train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
# ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
#     """UNUSED: Norm happens inside VisionTS model.
#
#     TODO: Confirm same norm method as B-MTGNN.
#     """
#     val_np = val_df.to_numpy()
#     test_np = test_df.to_numpy()
#     n_features = train_np.shape[1]
#     scale = np.ones(n_features)

#     for i in range(n_features):
#         # Compute scale factor as maximum absolute value of feature i in training data
#         scale[i] = np.max(np.abs(train_np[:, i]))

#         # Avoid division by zero
#         if scale[i] == 0:
#             scale[i] = 1.0
#             print(f"  Warning: Feature {i} has zero max value, using scale=1.0")

#         # Normalize each feature by its scale factor
#         train_np[:, i] = train_np[:, i] / scale[i]
#         val_np[:, i] = val_np[:, i] / scale[i]
#         test_np[:, i] = test_np[:, i] / scale[i]

#     print(f"Normalized {n_features} features using per-feature max scaling")
#     print(f"Scale factors - min: {scale.min():.6f}, max: {scale.max():.6f}, mean: {scale.mean():.6f}")

#     # Convert back to DataFrames with original index
#     train_df_norm = pd.DataFrame(train_np, index=train_df.index, columns=train_df.columns)
#     val_df_norm = pd.DataFrame(val_np, index=val_df.index, columns=val_df.columns)
#     test_df_norm = pd.DataFrame(test_np, index=test_df.index, columns=test_df.columns)

#     return train_df_norm, val_df_norm, test_df_norm, scale


