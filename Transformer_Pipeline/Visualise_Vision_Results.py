"""
Visualise_Vision_Results.py - Visualisation script for VisionTS model predictions.

This script mirrors the structure of Visualise_Results.py for consistency:
1. Loads predictions and ground truth from Results/
2. Inverse transforms PCA to get original feature space (if PCA was applied)
3. Generates forecast vs actual plots
4. Generates gap analysis plots (Threat vs Mitigation)
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Configuration
RESULTS_DIR = "Results"
DATA_DIR = os.path.join("..", "Processed_Data", "vision")

# Vision-specific file paths
PREDS_FILE = os.path.join(RESULTS_DIR, "vision_predictions.npy")
TRUES_FILE = os.path.join(RESULTS_DIR, "vision_ground_truth.npy")

# PCA and metadata paths
PCA_FILE = os.path.join(DATA_DIR, "pca_model.pkl")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.pkl")
FEATURE_NAMES_FILE = os.path.join(DATA_DIR, "feature_names.npy")


def get_col_index(name, all_cols):
    """Helper to find index of a specific column name."""
    try:
        return list(all_cols).index(name)
    except ValueError:
        print(f"WARNING: Column '{name}' not found in dataset!")
        print("Available columns snippet:", list(all_cols)[:5], "...")
        raise


def inverse_pca_transform(data, pca_model):
    """
    Inverse transform PCA-reduced data back to original feature space.

    Args:
        data: Array of shape (samples, time, n_components)
        pca_model: Fitted sklearn PCA object

    Returns:
        Array of shape (samples, time, n_original_features)
    """
    n_samples, n_time, n_components = data.shape

    # Reshape to 2D for inverse transform
    data_flat = data.reshape(-1, n_components)

    # Inverse transform
    data_original = pca_model.inverse_transform(data_flat)

    # Reshape back to 3D
    n_original_features = data_original.shape[1]
    return data_original.reshape(n_samples, n_time, n_original_features)


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

    # 2. Check if PCA was applied and inverse transform if needed
    pca_applied = False
    col_names = None

    if os.path.exists(PCA_FILE) and os.path.exists(FEATURE_NAMES_FILE):
        print("\n--- Inverse Transforming PCA ---")
        with open(PCA_FILE, "rb") as f:
            pca_model = pickle.load(f)

        col_names = np.load(FEATURE_NAMES_FILE, allow_pickle=True)

        print(f"PCA components: {pca_model.n_components_}")
        print(f"Original features: {len(col_names)}")

        # Inverse transform to original space
        preds_original = inverse_pca_transform(preds, pca_model)
        trues_original = inverse_pca_transform(trues, pca_model)

        print(f"Inverse transformed shape: {preds_original.shape}")
        pca_applied = True
    else:
        print("\nNo PCA model found - using predictions as-is")
        print("(Gap analysis with named features will not be available)")
        preds_original = preds
        trues_original = trues

        # Try to load feature names anyway
        if os.path.exists(FEATURE_NAMES_FILE):
            col_names = np.load(FEATURE_NAMES_FILE, allow_pickle=True)

    # 3. Plot A: Forecast vs Ground Truth (PCA Component or Feature)
    sample_idx = 0  # First test sample

    if pca_applied and col_names is not None:
        # Use original feature space with named columns
        THREAT_NAME = "DDoS-ALL"
        MITIGATION_NAME = "Solution_IDS/IPS_Papers"

        try:
            node_idx = get_col_index(THREAT_NAME, col_names)

            plt.figure(figsize=(12, 6))
            plt.plot(
                trues_original[sample_idx, :, node_idx],
                label="Ground Truth",
                marker="o",
                color="black",
                alpha=0.7,
            )
            plt.plot(
                preds_original[sample_idx, :, node_idx],
                label="VisionTS Forecast",
                marker="x",
                linestyle="--",
                color="blue",
            )

            plt.title(f"Forecast vs Actual: {THREAT_NAME}")
            plt.xlabel("Months Ahead")
            plt.ylabel("Intensity")
            plt.legend()
            plt.grid(True, linestyle=":", alpha=0.6)

            output_path_a = os.path.join(RESULTS_DIR, "vision_forecast_vs_actual.png")
            plt.savefig(output_path_a, dpi=300)
            plt.close()
            print(f"Saved: {output_path_a}")

        except ValueError:
            print(f"Could not find '{THREAT_NAME}' - plotting first feature instead")
            pca_applied = False  # Fall back to generic plot

    if not pca_applied:
        # Plot first few components/features
        n_features_to_plot = min(4, preds.shape[2])

        fig, axes = plt.subplots(n_features_to_plot, 1, figsize=(12, 3 * n_features_to_plot))
        if n_features_to_plot == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(trues[sample_idx, :, i], label="Ground Truth", linewidth=2)
            ax.plot(preds[sample_idx, :, i], label="VisionTS Forecast", linewidth=2, linestyle="--")
            ax.set_title(f"Component {i+1} - 36 Month Forecast")
            ax.set_xlabel("Months Ahead")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path_a = os.path.join(RESULTS_DIR, "vision_forecast_vs_actual.png")
        plt.savefig(output_path_a, dpi=300)
        plt.close()
        print(f"Saved: {output_path_a}")

    # 4. Plot B: Gap Analysis (Threat vs Mitigation) - only if we have original features
    if pca_applied and col_names is not None:
        THREAT_NAME = "DDoS-ALL"
        MITIGATION_NAME = "Solution_IDS/IPS_Papers"

        try:
            node_idx = get_col_index(THREAT_NAME, col_names)
            mitigation_idx = get_col_index(MITIGATION_NAME, col_names)

            threat_seq = preds_original[sample_idx, :, node_idx]
            mit_seq = preds_original[sample_idx, :, mitigation_idx]

            plt.figure(figsize=(12, 6))

            # Plot the main lines
            plt.plot(threat_seq, color="#d62728", linewidth=2, label=f"Threat: {THREAT_NAME}")
            plt.plot(mit_seq, color="#2ca02c", linewidth=2, label=f"Def: {MITIGATION_NAME}")

            # Fill the 'Risk Gap' (Where Threat > Mitigation)
            plt.fill_between(
                range(len(threat_seq)),
                threat_seq,
                mit_seq,
                where=(threat_seq > mit_seq),
                color="#d62728",
                alpha=0.2,
                hatch="//",
                label="Risk Gap",
            )

            # Fill the 'Safety Margin' (Where Mitigation > Threat)
            plt.fill_between(
                range(len(threat_seq)),
                threat_seq,
                mit_seq,
                where=(threat_seq <= mit_seq),
                color="#2ca02c",
                alpha=0.1,
                label="Safety Margin",
            )

            plt.title("Gap Analysis: Threat Intensity vs Mitigation Maturity (VisionTS)")
            plt.xlabel("Forecast Horizon (Months)")
            plt.ylabel("Normalized Score")
            plt.legend()
            plt.grid(True, linestyle=":", alpha=0.6)

            output_path_b = os.path.join(RESULTS_DIR, "vision_gap_analysis.png")
            plt.savefig(output_path_b, dpi=300)
            plt.close()
            print(f"Saved: {output_path_b}")

        except ValueError as e:
            print(f"Could not generate gap analysis: {e}")
    else:
        print("\nSkipping gap analysis (requires PCA inverse transform with named features)")


if __name__ == "__main__":
    visualise()