"""
Visualise_VisionPP_Results.py - Visualisation for VisionTS++ forecasting results.

Creates comparison charts between predictions and ground truth for selected features.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt


def get_col_index(name: str, all_cols: list) -> int:
    """Helper to find index of a specific column name."""
    try:
        return all_cols.index(name)
    except ValueError:
        print(f"WARNING: Column '{name}' not found!")
        print(f"Available columns: {all_cols[:10]}...")
        raise


def visualise():
    """Generate visualizations for VisionTS++ results."""
    # Configuration
    results_dir = SCRIPT_DIR / "Results"
    data_dir = SCRIPT_DIR.parent / "Processed_Data" / "visionpp"

    preds_file = results_dir / "visionpp_predictions.npy"
    trues_file = results_dir / "visionpp_ground_truth.npy"
    metadata_file = data_dir / "metadata.pkl"

    print(f"--- Loading Data from {results_dir} ---")

    if not preds_file.exists() or not trues_file.exists():
        print("Prediction files not found. Run Evaluate_VisionPP.py first.")
        return

    preds = np.load(preds_file)
    trues = np.load(trues_file)

    print(f"Predictions shape: {preds.shape} (Samples, Time, Features)")
    print(f"Ground truth shape: {trues.shape}")

    # Load metadata for feature names
    feature_names = None
    if metadata_file.exists():
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        feature_names = metadata.get("selected_feature_names") or metadata.get("feature_names")

        if metadata.get("feature_selection_applied"):
            print(f"\nMode: Feature Selection (named features)")
        elif metadata.get("pca_applied"):
            print(f"\nMode: PCA (latent space)")
        else:
            print(f"\nMode: Raw features")

    if feature_names is not None:
        print(f"Feature names loaded: {len(feature_names)} features")
    else:
        print("No feature names found - using indices")
        feature_names = [f"Feature_{i}" for i in range(preds.shape[2])]

    # Sample to visualize
    sample_idx = 0
    num_features = preds.shape[2]
    pred_len = preds.shape[1]

    # --- Plot 1: Multi-feature forecast comparison ---
    num_plots = min(4, num_features)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots))
    if num_plots == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(trues[sample_idx, :, i], label="Ground Truth", color="black", linewidth=2)
        ax.plot(preds[sample_idx, :, i], label="VisionTS++ Forecast", color="blue", linestyle="--", linewidth=2)

        title = f"{feature_names[i]} - {pred_len} Month Forecast"
        ax.set_title(title)
        ax.set_xlabel("Months Ahead")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = results_dir / "visionpp_forecast_vs_actual.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_path}")

    # --- Plot 2: Gap Analysis (if threat and defense features exist) ---
    # Look for threat-defense pairs
    threat_candidates = ["Ransomware-ALL", "Phishing-ALL", "Malware-ALL", "DDoS-ALL"]
    defense_candidates = ["Solution_ENCRYPTION_Papers", "Solution_CRYPTOGRAPHY_Papers",
                          "Solution_IDS/IPS_Papers", "Solution_RISK_ASSESSMENT_Papers"]

    threat_name = None
    defense_name = None

    for t in threat_candidates:
        if t in feature_names:
            threat_name = t
            break

    for d in defense_candidates:
        if d in feature_names:
            defense_name = d
            break

    if threat_name and defense_name:
        threat_idx = feature_names.index(threat_name)
        defense_idx = feature_names.index(defense_name)

        threat_seq = preds[sample_idx, :, threat_idx]
        defense_seq = preds[sample_idx, :, defense_idx]

        plt.figure(figsize=(12, 6))

        plt.plot(threat_seq, color="#d62728", linewidth=2, label=f"Threat: {threat_name}")
        plt.plot(defense_seq, color="#2ca02c", linewidth=2, label=f"Defense: {defense_name}")

        # Fill risk gap
        plt.fill_between(
            range(len(threat_seq)),
            threat_seq,
            defense_seq,
            where=(threat_seq > defense_seq),
            color="#d62728",
            alpha=0.2,
            hatch="//",
            label="Risk Gap",
        )

        # Fill safety margin
        plt.fill_between(
            range(len(threat_seq)),
            threat_seq,
            defense_seq,
            where=(threat_seq <= defense_seq),
            color="#2ca02c",
            alpha=0.1,
            label="Safety Margin",
        )

        plt.title(f"Gap Analysis: {threat_name} vs {defense_name}")
        plt.xlabel("Forecast Horizon (Months)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

        gap_path = results_dir / "visionpp_gap_analysis.png"
        plt.savefig(gap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {gap_path}")
    else:
        print("No threat-defense pair found for gap analysis")

    # --- Plot 3: Single feature detailed comparison ---
    # Use the first threat feature or first feature
    if threat_name:
        focus_idx = feature_names.index(threat_name)
        focus_name = threat_name
    else:
        focus_idx = 0
        focus_name = feature_names[0]

    plt.figure(figsize=(12, 6))
    plt.plot(
        trues[sample_idx, :, focus_idx],
        label="Ground Truth",
        marker="o",
        color="black",
        alpha=0.7,
    )
    plt.plot(
        preds[sample_idx, :, focus_idx],
        label="VisionTS++ Forecast",
        marker="x",
        linestyle="--",
        color="blue",
    )

    plt.title(f"Forecast vs Actual: {focus_name}")
    plt.xlabel("Months Ahead")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    single_path = results_dir / "visionpp_single_forecast.png"
    plt.savefig(single_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {single_path}")


if __name__ == "__main__":
    visualise()
