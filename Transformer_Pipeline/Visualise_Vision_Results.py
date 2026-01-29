"""
Visualise_Vision_Results.py - Visualisation script for VisionTS model predictions.

Supports two modes:
1. Feature Selection mode: Shows forecasts for named features (interpretable)
2. PCA mode: Shows forecasts for PCA components (latent space)
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Configuration
RESULTS_DIR = "Results"
DATA_DIR = os.path.join("..", "Processed_Data", "vision")

# Vision-specific file paths
PREDS_FILE = os.path.join(RESULTS_DIR, "vision_predictions.npy")
TRUES_FILE = os.path.join(RESULTS_DIR, "vision_ground_truth.npy")

# Metadata paths
METADATA_FILE = os.path.join(DATA_DIR, "metadata.pkl")
FEATURE_NAMES_FILE = os.path.join(DATA_DIR, "feature_names.npy")


def visualise():
    """Generate visualisation plots for VisionTS predictions."""

    # 1. Load Predictions
    print(f"--- Loading Data from {RESULTS_DIR} ---")
    if not os.path.exists(PREDS_FILE) or not os.path.exists(TRUES_FILE):
        raise FileNotFoundError(
            f"Prediction files not found at {PREDS_FILE}. Run Evaluate_Vision.py first."
        )

    preds = np.load(PREDS_FILE)
    trues = np.load(TRUES_FILE)
    print(f"Predictions shape: {preds.shape} (Samples, Time, Features)")
    print(f"Ground truth shape: {trues.shape}")

    # 2. Load metadata to determine mode
    feature_names = None
    use_feature_names = False

    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)

        if metadata.get("feature_selection_applied"):
            print("\nMode: Feature Selection (named features)")
            use_feature_names = True
        elif metadata.get("pca_applied"):
            print("\nMode: PCA (latent space)")
        else:
            print("\nMode: Raw features")
            use_feature_names = True

    # Load feature names if available
    if os.path.exists(FEATURE_NAMES_FILE):
        feature_names = np.load(FEATURE_NAMES_FILE, allow_pickle=True)
        print(f"Feature names loaded: {len(feature_names)} features")

    # 3. Plot: Forecast vs Ground Truth
    sample_idx = 0  # First test sample
    n_features = preds.shape[2]
    n_to_plot = min(4, n_features)

    _, axes = plt.subplots(n_to_plot, 1, figsize=(12, 3 * n_to_plot))
    if n_to_plot == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(trues[sample_idx, :, i], label="Ground Truth", linewidth=2, color="black")
        ax.plot(preds[sample_idx, :, i], label="VisionTS Forecast", linewidth=2, linestyle="--", color="blue")

        # Set title based on mode
        if use_feature_names and feature_names is not None and i < len(feature_names):
            title = f"{feature_names[i]} - 36 Month Forecast"
        else:
            title = f"Feature {i+1} - 36 Month Forecast"

        ax.set_title(title)
        ax.set_xlabel("Months Ahead")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, "vision_forecast_vs_actual.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    visualise()
