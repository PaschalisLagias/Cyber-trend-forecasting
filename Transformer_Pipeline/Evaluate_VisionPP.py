"""
Evaluate_VisionPP.py - Evaluation script for VisionTS++ model.

Computes metrics at multiple forecast horizons and saves results.
"""

from __future__ import annotations

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

    for h in horizons:
        # Get predictions up to horizon h
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

        # Correlation
        if np.std(p) > 0 and np.std(t) > 0:
            corr = np.corrcoef(p, t)[0, 1]
        else:
            corr = 0.0

        horizon_name = f"{h} Months" if h < 36 else f"Full ({h} Months)"
        results.append({
            "Horizon": horizon_name,
            "RSE": rse,
            "RAE": rae,
            "MAE": mae,
            "CORR": corr,
        })

    return pd.DataFrame(results)


def evaluate():
    """Run evaluation on the test set."""
    config = CyberVisionTSppConfig()
    data_dir = str(config.output_dir)

    print(f"--- Loading Test Data from {data_dir} ---")

    # Load test dataset
    test_dataset = CyberTrendImageDataset(data_dir=data_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    sample = test_dataset[0]
    context_len = sample["input"].shape[0]
    pred_len = sample["target"].shape[0]
    num_features = sample["input"].shape[1]

    print(f"Test samples: {len(test_dataset)}")
    print(f"Dimensions: context={context_len}, pred={pred_len}, features={num_features}")

    # Load model
    print(f"\n--- Loading VisionTS++ Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        model.load_state_dict(checkpoint["model_state_dict"])
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
            y_true = batch["target"]

            y_pred = model(x)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_true.numpy())

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

    # Save results
    results_dir = SCRIPT_DIR / "Results"
    results_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "visionpp_evaluation_results.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nSaved results table to {csv_path}")

    # Save raw predictions
    print(f"\n--- Saving Raw Predictions ---")
    np.save(results_dir / "visionpp_predictions.npy", preds)
    np.save(results_dir / "visionpp_ground_truth.npy", trues)
    print(f"Saved prediction arrays to {results_dir}/")

    # Also save to data directory for visualization
    np.save(config.output_dir / "predictions.npy", preds)
    np.save(config.output_dir / "ground_truth.npy", trues)
    print(f"Saved prediction arrays to {config.output_dir}/")

    # Print data preprocessing summary
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


if __name__ == "__main__":
    evaluate()
