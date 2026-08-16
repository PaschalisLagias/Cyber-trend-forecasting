"""
Clean Evaluation Table Generator

This script dynamically ingests prediction and ground truth arrays for both 
B-MTGNN and VisionTS++ models. It isolates the 26 target threat vectors and 
calculates forecasting error metrics using a Micro-Averaged approach. This 
protects against zero-variance blowouts on sparse threats and prevents 
dormant features from dragging down the Pearson correlation average.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Establish project root path
current_dir = Path.cwd()
project_root = current_dir
while not (project_root / "Data_Preparation").exists() and project_root.parent != project_root:
    project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from Config.Paths import (
    BMTGNN_PREDICTIONS,
    BMTGNN_HISTORY,
    BMTGNN_NAMES,
    BMTGNN_GLOBAL_DIR,
    VISION_GLOBAL_DIR,
    COMPARISON_PLOTS_DIR
)

# VisionTS++ specific paths
vision_results_dir = project_root / "Transformer_Pipeline" / "Results"
VISION_PREDICTIONS = vision_results_dir / "visionpp_predictions.npy"
VISION_GROUND_TRUTH = vision_results_dir / "visionpp_ground_truth.npy"
VISION_NAMES = vision_results_dir / "full_feature_names.npy"

# Overall comparative table directory
OVERALL_PLOTS_DIR = COMPARISON_PLOTS_DIR / "Overall"

TARGET_THREATS = [
    "Account Hijacking", "Adversarial Attack", "APT", "Backdoor", "Botnet",
    "Brute Force Attack", "Cryptojacking", "Data Poisoning", "DDoS", "Deepfake",
    "Disinformation", "DNS Spoofing", "Dropper", "Insider Threat", "IoT Device Attack",
    "Malware", "MITM", "Password Attack", "Phishing", "Ransomware",
    "Session Hijacking", "Supply Chain Attack", "Targeted Attack", "Trojan",
    "Vulnerability", "Zero-day"
]


def clean_string(s):
    """Sanitises strings for comparison matching."""
    s = str(s).lower().strip()
    s = s.replace('solution_', '').replace('_papers', '').replace('_mentions', '')
    s = s.replace('mentions-', '').replace('-all', '').replace('_', ' ')
    return s.title()


def get_target_indices(feature_names):
    """Locates the column indices for the 26 target threats."""
    indices = []
    for target in TARGET_THREATS:
        clean_target = clean_string(target).lower()
        if clean_target == "apt":
            clean_target = "advanced persistent threat"
            
        for i, raw_name in enumerate(feature_names):
            clean_col = clean_string(raw_name).lower()
            if clean_target == clean_col or clean_target in clean_col or clean_col in clean_target:
                if i not in indices:
                    indices.append(i)
                break
    return indices


def compute_metrics_micro(preds, trues):
    """
    Calculates RSE, RAE, MAE, and Pearson Correlation Coefficient using Micro-Averaging.
    Sums errors and variance across all 26 target features simultaneously to prevent 
    zero-variance mathematical blowouts on sparse features.

    Args:
        preds (np.ndarray): Predicted values array [steps, target_features].
        trues (np.ndarray): Ground truth values array [steps, target_features].

    Returns:
        tuple: (rse, rae, mae, corr)
    """
    diff = preds - trues
    
    # 1. Micro-Averaged Error Calculation
    sum_squared_diff = np.sum(diff ** 2)
    sum_abs_diff = np.sum(np.abs(diff))

    # Calculate feature-wise means for the variance denominator
    feature_means = np.mean(trues, axis=0)
    diff_mean = trues - feature_means
    
    sum_squared_var = np.sum(diff_mean ** 2)
    sum_abs_var = np.sum(np.abs(diff_mean))

    rse = np.sqrt(sum_squared_diff) / (np.sqrt(sum_squared_var) + 1e-5)
    rae = sum_abs_diff / (sum_abs_var + 1e-5)
    mae = np.mean(np.abs(diff))

    # 2. Masked Correlation Calculation
    # Calculate correlation feature-by-feature, ignoring dormant threats to prevent negative drag
    corrs = []
    for i in range(preds.shape[1]):
        p = preds[:, i]
        t = trues[:, i]
        if np.std(p) > 0 and np.std(t) > 0:
            c = np.corrcoef(p, t)[0, 1]
            if not np.isnan(c):
                corrs.append(c)

    corr = np.mean(corrs) if len(corrs) > 0 else 0.0

    return rse, rae, mae, corr


def evaluate_horizons(preds, trues, feature_names):
    """Slices arrays to target threats and evaluates predictions across horizons."""
    target_indices = get_target_indices(feature_names)
    
    preds_targets = preds[:, target_indices]
    trues_targets = trues[:, target_indices]

    horizons = [3, 6, 12, 24]
    max_steps = min(preds_targets.shape[0], trues_targets.shape[0])
    
    records = []
    for h in horizons:
        if h <= max_steps:
            p_sub = preds_targets[:h, :]
            t_sub = trues_targets[:h, :]
            rse, rae, mae, corr = compute_metrics_micro(p_sub, t_sub)
            records.append({
                "Horizon": f"{h} Months",
                "RSE": f"{rse:.3f}",
                "RAE": f"{rae:.3f}",
                "MAE": f"{mae:.3f}",
                "CORR": f"{corr:.3f}"
            })

    rse_all, rae_all, mae_all, corr_all = compute_metrics_micro(
        preds_targets[:max_steps, :], 
        trues_targets[:max_steps, :]
    )
    
    records.append({
        "Horizon": "Overall",
        "RSE": f"{rse_all:.3f}",
        "RAE": f"{rae_all:.3f}",
        "MAE": f"{mae_all:.3f}",
        "CORR": f"{corr_all:.3f}"
    })

    return pd.DataFrame(records)


def render_formatted_table(df, title_text, output_path, col_widths=None, show_footnote=True):
    """Renders a DataFrame into a formatted PNG table image."""
    num_rows = len(df)
    fig_height = num_rows * 0.55 + 1.8
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")

    ax.text(
        0.5, 0.92, title_text,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="bottom"
    )

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    overall_row_idx = None
    for idx, row in df.iterrows():
        if str(row.iloc[0]).strip().lower() == "overall":
            overall_row_idx = idx + 1
            break

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor("gray")

        if row == 0:
            cell.set_text_props(weight="bold", color="black")
            cell.set_facecolor("#e0e0e0")
        elif overall_row_idx is not None and row == overall_row_idx:
            cell.set_text_props(weight="bold", color="black")
            cell.set_facecolor("#d9d9d9")
            cell.set_linewidth(1.8)
            cell.set_edgecolor("black")
        else:
            if row % 2 == 0:
                cell.set_facecolor("#f9f9f9")
            else:
                cell.set_facecolor("white")

    if show_footnote:
        ax.text(
            0.02, 0.04,
            "*Note: Overall represents the 36-month forecast period.",
            transform=ax.transAxes,
            fontsize=10,
            fontstyle="italic",
            ha="left",
            va="top"
        )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def generate_all_tables():
    """Executes table generation pipeline for both architectures."""
    print("Evaluating B-MTGNN arrays...")
    try:
        b_preds = np.load(BMTGNN_PREDICTIONS).squeeze()
        b_trues = np.load(BMTGNN_HISTORY).squeeze()
        b_names = np.load(BMTGNN_NAMES, allow_pickle=True).tolist()
        
        if b_preds.ndim == 3:
            b_preds = b_preds[-1, :, :]
        if b_trues.ndim == 3:
            b_trues = b_trues[-1, :, :]

        if b_trues.shape[0] > b_preds.shape[0]:
            b_trues = b_trues[-b_preds.shape[0]:, :]
            
        b_df = evaluate_horizons(b_preds, b_trues, b_names)
    except Exception as e:
        print(f"Error loading B-MTGNN arrays: {e}")
        return

    b_out_path = BMTGNN_GLOBAL_DIR / "BMTGNN_Eval_Table.png"
    render_formatted_table(
        df=b_df,
        title_text="B-MTGNN: Global Metrics",
        output_path=b_out_path,
        show_footnote=True
    )
    print(f"Saved: {b_out_path}")

    print("Evaluating VisionTS++ arrays...")
    try:
        v_preds = np.load(VISION_PREDICTIONS).squeeze()
        v_trues = np.load(VISION_GROUND_TRUTH).squeeze()
        v_names = np.load(VISION_NAMES, allow_pickle=True).tolist()
        
        if v_preds.ndim == 3:
            v_preds = v_preds[-1, :, :]
        if v_trues.ndim == 3:
            v_trues = v_trues[-1, :, :]

        if v_trues.shape[0] > v_preds.shape[0]:
            v_trues = v_trues[-v_preds.shape[0]:, :]
            
        v_df = evaluate_horizons(v_preds, v_trues, v_names)
    except Exception as e:
        print(f"Error loading VisionTS++ arrays: {e}")
        return

    v_out_path = VISION_GLOBAL_DIR / "Vision_Eval_Table.png"
    render_formatted_table(
        df=v_df,
        title_text="VisionTS++: Global Metrics",
        output_path=v_out_path,
        show_footnote=True
    )
    print(f"Saved: {v_out_path}")

    print("Generating Comparative Summary Table...")
    b_overall = b_df[b_df["Horizon"] == "Overall"].iloc[0]
    v_overall = v_df[v_df["Horizon"] == "Overall"].iloc[0]

    comp_df = pd.DataFrame([
        {"Model": "B-MTGNN", "RSE": b_overall["RSE"], "RAE": b_overall["RAE"]},
        {"Model": "VisionTS++", "RSE": v_overall["RSE"], "RAE": v_overall["RAE"]}
    ])

    comp_out_path = OVERALL_PLOTS_DIR / "Table_9_Comparative_Evaluation.png"
    render_formatted_table(
        df=comp_df,
        title_text="Comparative Evaluation",
        output_path=comp_out_path,
        col_widths=[0.3, 0.35, 0.35],
        show_footnote=False
    )
    print(f"Saved: {comp_out_path}")


if __name__ == "__main__":
    generate_all_tables()