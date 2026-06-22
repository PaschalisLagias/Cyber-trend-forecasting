"""
Computes metrics at multiple forecast horizons and saves results.
"""

from __future__ import annotations

import argparse
import os
import pickle
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
from Cyber_Trend_Image_Dataset import CyberTrendImageDataset
from Cyber_Trend_VisionPP_Config import CyberVisionTSppConfig
from torch.utils.data import DataLoader

from Models.VisionTSpp_Wrapper import create_visiontspp_model


def compute_metrics_per_horizon(preds: np.ndarray, trues: np.ndarray, horizons: list[int]) -> pd.DataFrame:
    """
    Compute RSE, RAE, MAE, and CORR for each forecast horizon.

    Args:
        preds: Predictions array of shape (samples, pred_len, features)
        trues: Ground truth array of shape (samples, pred_len, features)
        horizons: List of horizon endpoints to evaluate (e.g., [3, 6, 12, 24, 36])

    Returns:
        DataFrame with metrics for each horizon
    """
    results = []
    max_h = preds.shape[1]

    # --- 1. Detailed Horizon Breakdown ---
    for h in horizons:
        if h < max_h:
            p = preds[:, :h, :].flatten()
            t = trues[:, :h, :].flatten()

            # RSE
            mse = np.mean((p - t) ** 2)
            var = np.var(t)
            rse = np.sqrt(mse / (var + 1e-7))

            # RAE
            mae = np.mean(np.abs(p - t))
            mean_abs_dev = np.mean(np.abs(t - np.mean(t)))
            rae = mae / (mean_abs_dev + 1e-7)

            # CORRELATION
            if np.std(p) > 0 and np.std(t) > 0:
                corr = np.corrcoef(p, t)[0, 1]
            else:
                corr = 0.0

            horizon_name = f"{h} Months"
            results.append({
                "Horizon": horizon_name,
                "RSE": rse,
                "RAE": rae,
                "MAE": mae,
                "CORR": corr,
            })

    # --- 2. Overall Summary  ---
    p_all = preds[:, :max_h, :].flatten()
    t_all = trues[:, :max_h, :].flatten()

    mse_all = np.mean((p_all - t_all) ** 2)
    var_all = np.var(t_all)
    rse_all = np.sqrt(mse_all / (var_all + 1e-7))

    mae_all = np.mean(np.abs(p_all - t_all))
    mean_abs_dev_all = np.mean(np.abs(t_all - np.mean(t_all)))
    rae_all = mae_all / (mean_abs_dev_all + 1e-7)

    if np.std(p_all) > 0 and np.std(t_all) > 0:
        corr_all = np.corrcoef(p_all, t_all)[0, 1]
    else:
        corr_all = 0.0

    results.append({
        "Horizon": "Overall",
        "RSE": rse_all,
        "RAE": rae_all,
        "MAE": mae_all,
        "CORR": corr_all,
    })

    return pd.DataFrame(results)

    return pd.DataFrame(results)


def evaluate(dataset: str = "cyber_trend", use_custom_vit: bool = False,
             customvit_layers: int | None = None,
             customvit_heads: int | None = None,
             customvit_dim: int | None = None):
    """Run evaluation on the test set for the given dataset."""
    config = CyberVisionTSppConfig()
    overrides: dict = {"dataset": dataset, "use_custom_vit": use_custom_vit}
    if customvit_layers is not None:
        overrides["customvit_layers"] = customvit_layers
    if customvit_heads is not None:
        overrides["customvit_heads"] = customvit_heads
    if customvit_dim is not None:
        overrides["customvit_dim"] = customvit_dim
    config.update_from_dict(overrides)
    data_dir = str(config.output_dir)

    print(f"--- Loading Test Data from {data_dir} ---")

    # Load test dataset (detrend if the config says so; same as Train_VisionPP)
    detrend = getattr(config, "detrend", False)
    test_dataset = CyberTrendImageDataset(data_dir=data_dir, split="test", detrend=detrend)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    sample = test_dataset[0]
    context_len = sample["input"].shape[0]
    pred_len = sample["target"].shape[0]
    num_features = sample["input"].shape[1]

    print(f"Test samples: {len(test_dataset)}")
    print(f"Dimensions: context={context_len}, pred={pred_len}, features={num_features}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.use_custom_vit:
        print(f"\n--- Loading Custom ViT ---")
        from Models.CustomViT_Wrapper import CustomViTModel
        vit_config = {
            "customvit_layers": config.customvit_layers,
            "customvit_heads": config.customvit_heads,
            "customvit_dim": config.customvit_dim,
            "customvit_patch_size": config.customvit_patch_size,
            "customvit_dropout": config.customvit_dropout,
            "use_aux_direction": config.use_aux_direction,
        }
        model = CustomViTModel(
            config=vit_config,
            context_len=context_len,
            pred_len=pred_len,
            n_features=num_features,
            finetune_type=config.finetune_type,
            load_pretrained=False,  # weights loaded from the trained checkpoint below
        ).to(device)
    else:
        print(f"\n--- Loading VisionTS++ Model ---")
        model_config = {
            "model_arch": config.model_arch,
            "finetune_type": config.finetune_type,
            "ckpt_dir": str(config.ckpt_dir),
            "load_pretrained": config.load_pretrained,
            "periodicity": config.periodicity,
            "norm_const": config.norm_const,
            "align_const": config.align_const,
            "interpolation": config.interpolation,
            "quantile": config.quantile,
            "color": config.color,
        }
        model = create_visiontspp_model(
            config=model_config,
            context_len=context_len,
            pred_len=pred_len,
            num_features=num_features,
            device=str(device),
        )

    # Load checkpoint
    checkpoint_path = config.checkpoint_dir / "visiontspp_best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False) # Added strict=False to bypass missing quantile heads
        print(f"Loaded weights from {checkpoint_path}")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Checkpoint val_loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}")
        print("Running with pretrained weights only.")

    # Run inference
    print(f"\n--- Running Inference ---")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["input"].to(device)
            y_true = batch["target"].to(device)

            y_pred = model(x)

            # Reconstruct in original space if the dataset was detrended.
            if detrend and "trend_slope" in batch:
                slope = batch["trend_slope"].to(device).unsqueeze(1)        # (B, 1, V)
                intercept = batch["trend_intercept"].to(device).unsqueeze(1)  # (B, 1, V)
                t_pred = torch.arange(
                    context_len, context_len + pred_len,
                    device=device, dtype=slope.dtype,
                ).view(1, -1, 1)
                pred_trend = slope * t_pred + intercept
                y_pred = y_pred + pred_trend
                y_true = y_true + pred_trend

            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_targets, axis=0)

    print(f"Predictions shape: {preds.shape}")
    print(f"Ground truth shape: {trues.shape}")

    # Compute metrics at different horizons
    print(f"\n--- Calculating Metrics per Horizon ---")
    horizons = [3, 6, 12, 24, pred_len]
    metrics_df = compute_metrics_per_horizon(preds, trues, horizons)

    # Display results
    print(f"\n{'=' * 60}")
    print("Evaluation Results")
    print(f"{'=' * 60}")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Save results.
    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "visionpp_evaluation_results.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nSaved results table to {csv_path}")

    print(f"\n--- Saving Raw Predictions ---")
    np.save(results_dir / "visionpp_predictions.npy", preds)
    np.save(results_dir / "visionpp_ground_truth.npy", trues)
    print(f"Saved prediction arrays to {results_dir}/")

    np.save(config.output_dir / "predictions.npy", preds)
    np.save(config.output_dir / "ground_truth.npy", trues)
    print(f"Saved prediction arrays to {config.output_dir}/")

    print(f"\n--- Data Preprocessing Summary ---")
    metadata_path = Path(data_dir) / "metadata.pkl"
    if metadata_path.exists():
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        print(f"Original features: {metadata.get('original_num_features', 'N/A')}")
        if metadata.get('feature_selection_applied'):
            print(f"Feature selection: {metadata.get('num_features', 'N/A')} features selected")
        elif metadata.get('pca_applied'):
            print(f"PCA components: {metadata.get('pca_n_components', 'N/A')}")
            pca_var = metadata.get('pca_explained_variance')
            if pca_var is not None:
                print(f"PCA variance: {pca_var:.1%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VisionTS++ on a preprocessed dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cyber_trend",
        choices=["cyber_trend", "csis", "mark3"],
        help="Dataset to evaluate (default: cyber_trend). Drives input data dir and output Results subdir.",
    )
    parser.add_argument(
        "--use-custom-vit",
        action="store_true",
        help="Build and evaluate the CustomViT instead of VisionTS++ (matches the model trained by Train_VisionPP --use_custom_vit).",
    )
    parser.add_argument("--customvit-layers", type=int, help="Override CustomViT n_layers (must match the trained model).")
    parser.add_argument("--customvit-heads", type=int, help="Override CustomViT n_heads (must match the trained model).")
    parser.add_argument("--customvit-dim", type=int, help="Override CustomViT embed_dim (must match the trained model).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        dataset=args.dataset, use_custom_vit=args.use_custom_vit,
        customvit_layers=args.customvit_layers,
        customvit_heads=args.customvit_heads,
        customvit_dim=args.customvit_dim,
    )
