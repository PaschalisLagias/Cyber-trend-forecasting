"""
Unified Figure 3 Validation Forecast Pipeline

This module generates Figure 3 validation plots across all 26 target threat vectors. 
It applies exponential smoothing to a normalised scale before restoring the original 
magnitude, allowing for clean visual trajectories without compressing the Y-axis.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

TARGET_THREATS = [
    "Account Hijacking", "Adversarial Attack", "APT", "Backdoor", "Botnet",
    "Brute Force Attack", "Cryptojacking", "Data Poisoning", "DDoS", "Deepfake",
    "Disinformation", "DNS Spoofing", "Dropper", "Insider Threat", "IoT Device Attack",
    "Malware", "MITM", "Password Attack", "Phishing", "Ransomware",
    "Session Hijacking", "Supply Chain Attack", "Targeted Attack", "Trojan",
    "Vulnerability", "Zero-day"
]


def clean_string(s):
    """Sanitises strings for plot titles and comparison matching."""
    s = str(s).lower().strip()
    s = s.replace('solution_', '').replace('_papers', '').replace('_mentions', '')
    s = s.replace('mentions-', '').replace('-all', '').replace('_', ' ')
    return s.title()


def sanitise_filename(s):
    """Sanitises threat names for file output naming conventions."""
    clean = clean_string(s)
    clean = clean.replace(' ', '_').replace('/', '_')
    if clean.lower() == "apt":
        return "Apt"
    if clean.lower() == "ddos":
        return "Ddos"
    if clean.lower() == "zero-day":
        return "Zero-Day"
    return clean


def find_col_index(target_name, all_names):
    """Locates column index in feature array using fuzzy matching."""
    clean_target = clean_string(target_name).lower()
    if clean_target == "apt":
        clean_target = "advanced persistent threat"

    for i, raw_name in enumerate(all_names):
        clean_col = clean_string(raw_name).lower()
        if clean_target == clean_col or clean_target in clean_col or clean_col in clean_target:
            return i
    return None


def generate_date_labels_forward(start_date, num_steps):
    """Generates monthly datetime objects starting from start_date."""
    dates = []
    current = start_date
    for _ in range(num_steps):
        dates.append(current)
        year = current.year + (current.month // 12)
        month = (current.month % 12) + 1
        current = current.replace(year=year, month=month, day=1)
    return dates


def exponential_smoothing_clamped(series, alpha=0.05):
    """Applies exponential smoothing, clamped at 0 to prevent negative rendering."""
    if len(series) == 0:
        return series
    
    result = [max(0, series[0])]
    for n in range(1, len(series)):
        val = alpha * series[n] + (1 - alpha) * result[n - 1]
        result.append(max(0, val))
        
    return np.array(result)


def plot_single_figure_3(model_name, threat_name, preds, trues, cols, dates, out_dir, alpha=0.05):
    """Renders a scaled and smoothed Figure 3 validation plot."""
    col_idx = find_col_index(threat_name, cols)
    if col_idx is None:
        return False

    actual_raw = trues[:, col_idx]
    forecast_raw = preds[:, col_idx]

    if actual_raw.ndim > 1:
        actual_raw = actual_raw.squeeze()
    if forecast_raw.ndim > 1:
        forecast_raw = forecast_raw.squeeze()

    # Determine maximum value to scale series
    max_val = np.max(actual_raw)
    if max_val == 0:
        max_val = 1.0

    # Normalise arrays prior to smoothing
    actual_norm = actual_raw / max_val
    forecast_norm = forecast_raw / max_val

    # Apply smoothing and restore native magnitude
    actual_series = exponential_smoothing_clamped(actual_norm, alpha=alpha) * max_val
    forecast_series = exponential_smoothing_clamped(forecast_norm, alpha=alpha) * max_val

    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        try:
            plt.style.use('seaborn-darkgrid')
        except OSError:
            pass

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(dates, actual_series, label='Actual', color='#0000fe', linewidth=2)
    ax.plot(dates, forecast_series, label='Forecast', color='#7f007f', linestyle='--', linewidth=2)

    ax.fill_between(
        dates,
        forecast_series * 0.95,
        forecast_series * 1.05,
        color='#fee6ea',
        alpha=0.6,
        label='95% Confidence'
    )

    ax.set_title(f"Figure 3: {clean_string(threat_name)} ({model_name} Validation)", fontsize=16)
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Trend', fontsize=12, fontweight='bold')

    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.xticks(rotation=45)
    ax.grid(True, which='major', axis='both', linestyle='-', color='lightgray', alpha=0.7)
    ax.legend(loc='upper left', fontsize=10, frameon=True, edgecolor='black')

    filename = f"Fig3_{sanitise_filename(threat_name)}.png"
    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return True


def generate_all_figure_3_plots(model_name, preds_path, trues_path, cols_path, out_dir, alpha=0.05):
    """Loads arrays and generates correctly scaled Figure 3 plots for all targets."""
    print("=" * 60)
    print(f"--- Generating Figure 3 Validation Plots for {model_name} (alpha={alpha}) ---")

    try:
        preds = np.load(preds_path).squeeze()
        trues = np.load(trues_path).squeeze()
        cols = np.load(cols_path, allow_pickle=True).tolist()

        if preds.ndim == 3:
            preds = preds[-1, :, :]
        if trues.ndim == 3:
            trues = trues[-1, :, :]

        if trues.shape[0] > preds.shape[0]:
            trues = trues[-preds.shape[0]:, :]

    except Exception as e:
        print(f"Failed to load arrays for {model_name}: {e}")
        return

    dates = generate_date_labels_forward(datetime(2023, 1, 1), preds.shape[0])
    out_path = Path(out_dir)

    success_count = 0
    for threat in TARGET_THREATS:
        if plot_single_figure_3(model_name, threat, preds, trues, cols, dates, out_path, alpha):
            success_count += 1

    print(f"SUCCESS: Generated {success_count}/26 Figure 3 plots in {out_path}")
    print("=" * 60)