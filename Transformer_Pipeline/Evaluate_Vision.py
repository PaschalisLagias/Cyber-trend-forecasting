"""
Evaluate_Vision.py - Evaluation script for VisionTS model on cyber threat data.

This script mirrors the structure of Evaluate_Graph.py for consistency:
1. Loads the trained VisionTS model
2. Runs inference on the test set
3. Computes metrics at multiple horizons
4. Saves predictions and results for visualization
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add script directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from Cyber_Trend_Image_Dataset import CyberTrendImageDataset
from Cyber_Trend_Vision_Config import CyberVisionTSConfig
from Models.VisionTS_Wrapper import create_visionts_model
from Utils.Metrics import calculate_all_metrics

# Configuration
RESULTS_DIR = "Results"
DATA_DIR = str(SCRIPT_DIR.parent / "Processed_Data" / "vision")


def evaluate(data_dir: str = None, checkpoint_path: str = None) -> dict:
    """
    Evaluate VisionTS model on test set.

    Args:
        data_dir: Path to preprocessed data directory
        checkpoint_path: Path to model checkpoint (optional, uses default if not provided)

    Returns:
        Dictionary of evaluation metrics
    """
    # 1. Setup Environment
    config = CyberVisionTSConfig()
    device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Use provided data_dir or default
    data_dir = data_dir or DATA_DIR
    print(f"--- Loading Test Data from {data_dir} ---")

    # 2. Load Test Data
    test_dataset = CyberTrendImageDataset(data_dir=data_dir, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Get dimensions from dataset
    sample = test_dataset[0]
    context_len = sample["input"].shape[0]
    pred_len = sample["target"].shape[0]
    num_features = sample["input"].shape[1]

    print(f"Test samples: {len(test_dataset)}")
    print(f"Dimensions: context={context_len}, pred={pred_len}, features={num_features}")

    # 3. Load Model
    print("\n--- Loading VisionTS Model ---")

    model_config = {
        "model_arch": config.model_arch,
        "finetune_type": config.finetune_type,
        "ckpt_dir": str(config.ckpt_dir),
        "load_pretrained": config.load_pretrained,
        "periodicity": config.periodicity,
        "norm_const": config.norm_const,
        "align_const": config.align_const,
        "interpolation": config.interpolation,
    }

    model = create_visionts_model(
        config=model_config,
        context_len=context_len,
        pred_len=pred_len,
        num_features=num_features,
        device=str(device),
    )

    # Load trained weights if checkpoint exists
    if checkpoint_path is None:
        checkpoint_path = Path(config.checkpoint_dir) / "vision_best.pt"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded weights from {checkpoint_path}")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Checkpoint val_loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        print("Using pretrained MAE weights (zero-shot mode)")

    model.eval()

    # 4. Inference Loop
    print("\n--- Running Inference ---")
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["input"].to(device)
            y_true = batch["target"].to(device)

            # Forward pass
            y_pred = model(x)

            # Move to CPU and numpy
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y_true.cpu().numpy())

    # Concatenate all batches -> Shape: (Total_Samples, Pred_Len, Features)
    preds_array = np.concatenate(all_preds, axis=0)
    trues_array = np.concatenate(all_trues, axis=0)

    print(f"Predictions shape: {preds_array.shape}")
    print(f"Ground truth shape: {trues_array.shape}")

    # 5. Generate Metrics Table
    print("\n--- Calculating Metrics per Horizon ---")

    # Determine horizons based on prediction length
    max_horizon = preds_array.shape[1]
    if max_horizon >= 24:
        horizons = [3, 6, 12, 24]
    elif max_horizon >= 12:
        horizons = [3, 6, 12]
    elif max_horizon >= 6:
        horizons = [3, 6]
    else:
        horizons = [max_horizon]

    # Filter horizons to those within prediction length
    horizons = [h for h in horizons if h <= max_horizon]

    results_list = []
    for h in horizons:
        # Slice data up to horizon h
        p_slice = preds_array[:, :h, :]
        t_slice = trues_array[:, :h, :]

        metrics = calculate_all_metrics(p_slice, t_slice)
        metrics["Horizon"] = f"{h} Months"
        results_list.append(metrics)

    # Also compute full-horizon metrics
    full_metrics = calculate_all_metrics(preds_array, trues_array)
    full_metrics["Horizon"] = f"Full ({max_horizon} Months)"
    results_list.append(full_metrics)

    # Create DataFrame and Save
    results_df = pd.DataFrame(results_list)
    # Reorder columns
    results_df = results_df[["Horizon", "RSE", "RAE", "MAE", "CORR"]]

    csv_path = os.path.join(RESULTS_DIR, "vision_evaluation_results.csv")
    results_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nSaved results table to {csv_path}")

    # 6. Save Raw Predictions for Visualisation
    print("\n--- Saving Raw Predictions ---")
    np.save(os.path.join(RESULTS_DIR, "vision_predictions.npy"), preds_array)
    np.save(os.path.join(RESULTS_DIR, "vision_ground_truth.npy"), trues_array)
    print(f"Saved prediction arrays to {RESULTS_DIR}/")

    # Also save to the data directory for consistency with Train_Vision.py
    np.save(os.path.join(data_dir, "predictions.npy"), preds_array)
    np.save(os.path.join(data_dir, "ground_truth.npy"), trues_array)
    print(f"Saved prediction arrays to {data_dir}/")

    # 7. Load and display metadata if available
    metadata_path = os.path.join(data_dir, "metadata.pkl")
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        print("\n--- Data Preprocessing Summary ---")
        print(f"Original features: {metadata.get('original_num_features', 'N/A')}")
        print(f"PCA components:    {metadata.get('pca_n_components', 'N/A')}")
        print(f"PCA variance:      {metadata.get('pca_explained_variance', 0):.1%}")

    return {
        "results_df": results_df,
        "predictions": preds_array,
        "ground_truth": trues_array,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate VisionTS model")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to preprocessed data")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()

    evaluate(data_dir=args.data_dir, checkpoint_path=args.checkpoint)
