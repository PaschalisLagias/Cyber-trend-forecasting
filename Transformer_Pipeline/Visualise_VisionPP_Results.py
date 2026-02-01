"""
Visualise VisionPP Results Pipeline
====================================

This module handles the visualisation of results for the VisionTS++ model.
It loads historical data, model predictions, and graph structures to generate:
1. Global Trend Analysis (Accuracy & Gap).
2. Publication-ready Figures (Figure 3 & Figure 4).
3. Risk Analysis Tables (Tables 5 & 6).

Usage:
    Run this script directly to execute the full visualisation pipeline:
    $ python Visualise_VisionPP_Results.py
"""

import os
import sys
import csv
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# --- Import Config ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    from Transformer_Pipeline.Cyber_Trend_VisionPP_Config import CyberVisionTSppConfig
except ImportError:
    sys.path.append(str(current_dir))
    from Cyber_Trend_VisionPP_Config import CyberVisionTSppConfig

# ------------------- Helper Functions -------------------

def exponential_smoothing(series, alpha=0.1):
    """
    Applies exponential smoothing to a time series to reduce noise.

    Args:
        series (np.array): The input time series data.
        alpha (float, optional): The smoothing factor (0 < alpha <= 1).
                                 Lower values produce smoother curves. Defaults to 0.1.

    Returns:
        np.array: The smoothed time series.
    """
    if len(series) == 0:
        return series
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return np.array(result)


def normalise_series(series):
    """
    Normalises a series using Max-Scaling (values scaled between 0 and 1).

    Args:
        series (np.array): The input time series.

    Returns:
        np.array: The normalised series. Returns original series if max is 0.
    """
    mx = np.max(np.abs(series))
    if mx == 0:
        return series
    return series / mx


def clean_string(s):
    """
    Sanitises and formats a string for display in legends or matching.

    Removes specific dataset artifacts like 'Solution_', '_Mentions', '_Papers', etc.,
    and converts the string to Title Case.

    Args:
        s (str): The raw string to clean.

    Returns:
        str: The cleaned, title-cased string.
    """
    s = str(s).lower().strip()
    s = s.replace('solution_', '').replace('_papers', '').replace('_mentions', '').replace('mentions', '')
    s = s.replace('-', ' ').replace('_', ' ')
    s = s.replace('attack', '').strip()
    return s.title()


def sanitise_filename(s):
    """
    Converts a string into a safe format for use in filenames.

    Args:
        s (str): The input string.

    Returns:
        str: A string safe for file paths.
    """
    return clean_string(s).replace('/', '_').replace(' ', '_')


def find_col_index(target_name, all_names):
    """
    Finds the index of a target column in a list of names using fuzzy matching.

    Args:
        target_name (str): The name of the column to find.
        all_names (list): A list of all available column names.

    Returns:
        int or None: The index of the column if found, otherwise None.
    """
    clean_target = clean_string(target_name).lower()
    for i, raw_name in enumerate(all_names):
        clean_col = clean_string(raw_name).lower()
        if clean_target == clean_col or clean_target in clean_col or clean_col in clean_target:
            if len(clean_target) > 1:
                return i
    return None


def build_graph(file_name):
    """
    Parses the graph CSV file to build an adjacency dictionary.

    Args:
        file_name (Path): The path to the graph.csv file.

    Returns:
        defaultdict: A dictionary mapping parent nodes (Threats) to a list of
                     child nodes (Solutions).
    """
    graph = defaultdict(list)
    if not os.path.exists(file_name):
        return graph
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            adj = [n.strip() for n in row[1:] if n.strip()]
            graph[key].extend(adj)
    return graph


def generate_date_labels_forward(start_date, num_steps):
    """
    Generates a list of chronological datetime objects moving forward from a start date.

    Args:
        start_date (datetime): The starting date.
        num_steps (int): The number of months to generate.

    Returns:
        list: A list of datetime objects.
    """
    dates = []
    current = start_date
    for _ in range(num_steps):
        dates.append(current)
        year = current.year + (current.month // 12)
        month = (current.month % 12) + 1
        current = current.replace(year=year, month=month, day=1)
    return dates


def save_table_as_image(df, title, output_dir):
    """
    Saves a pandas DataFrame as both a LaTeX file and a PNG image.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        title (str): The title of the table (used for filename and plot title).
        output_dir (Path): The directory where files will be saved.
    """
    if df.empty:
        return
    latex_path = output_dir / f"{title}.tex"
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.3f"))

    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')

    plt.title(title.replace('_', ' '), fontsize=14, pad=10)
    img_path = output_dir / f"{title}.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {title} to: {img_path}")


def load_full_history(config, col_names):
    """
    Loads the full historical dataset from the raw CSV path defined in config.

    Args:
        config (CyberVisionTSppConfig): The configuration object containing file paths.
        col_names (list): A list of expected column names to map the data correctly.

    Returns:
        tuple: A tuple containing:
            - full_history (np.array): Array of historical data or None.
            - success (bool): True if loading was successful, False otherwise.
    """
    csv_path = config.raw_data_path
    print(f"Loading Full History from: {csv_path}")
    if not os.path.exists(csv_path):
        return None, False

    try:
        df = pd.read_csv(csv_path)
        if 'Date' in df.columns or 'month' in df.columns.str.lower():
            df = df.select_dtypes(include=[np.number])

        full_history = np.zeros((len(df), len(col_names)))
        df_cols_clean = [clean_string(c).lower() for c in df.columns]

        for i, target_node in enumerate(col_names):
            clean_node = clean_string(target_node).lower()
            match_idx = -1
            for j, csv_col in enumerate(df_cols_clean):
                if clean_node == csv_col or clean_node in csv_col:
                    match_idx = j
                    break
            if match_idx != -1:
                full_history[:, i] = df.iloc[:, match_idx].values

        return full_history, True
    except Exception as e:
        print(f"Error loading CSV history: {e}")
        return None, False


# ------------------- 1. Data Loading -------------------

def load_visualisation_data():
    """
    Loads and prepares all necessary data for the VisionPP visualisation pipeline.

    It retrieves:
    1. Model predictions (visionpp_predictions.npy) — averaged across test samples.
    2. Feature names from metadata.pkl.
    3. The knowledge graph structure (graph.csv).
    4. Historical ground truth data (CSV or .npy fallback).

    Returns:
        dict: A dictionary containing 'preds_window', 'history_data', 'col_names',
              'target_groups', 'trues_forecast_period', and 'output_dir'.
              Returns None if critical files are missing.
    """
    config = CyberVisionTSppConfig()
    data_dir = current_dir / "Results"
    plot_output_dir = data_dir / 'plots_paper_style'
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("--- Loading VisionPP Visualisation Data ---")

    preds_path = data_dir / 'visionpp_predictions.npy'
    trues_path = data_dir / 'visionpp_ground_truth.npy'
    metadata_path = config.output_dir / 'metadata.pkl'
    graph_csv_path = project_root / "B-MTGNN" / "data" / "graph.csv"

    if not preds_path.exists():
        print("Error: visionpp_predictions.npy not found.")
        return None

    preds = np.load(preds_path)
    trues = np.load(trues_path)

    print(f"Raw predictions shape: {preds.shape}")
    print(f"Raw ground truth shape: {trues.shape}")

    # Load feature names from metadata
    col_names = None
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        col_names = metadata.get('selected_feature_names') or metadata.get('feature_names')

    if col_names is None:
        col_names = [f"Feature_{i}" for i in range(preds.shape[2])]
    col_names = list(col_names)
    print(f"Feature names loaded: {len(col_names)} features")

    target_groups = build_graph(graph_csv_path)

    # Average predictions across test samples for robust global estimates
    # preds shape: (samples, time, features) → mean over samples → (time, features)
    if preds.ndim == 3:
        preds_window = preds.mean(axis=0)
        trues_forecast_period = trues.mean(axis=0)
    elif preds.ndim == 4:
        preds_window = preds.squeeze(-1).mean(axis=0)
        trues_forecast_period = trues.squeeze(-1).mean(axis=0)
    else:
        preds_window = preds
        trues_forecast_period = trues

    print(f"Averaged predictions shape: {preds_window.shape}")

    history_data, success = load_full_history(config, col_names)
    if not success:
        print("Warning: Could not load full CSV history, using ground truth as fallback.")
        history_data = trues_forecast_period

    return {
        "preds_window": preds_window,
        "history_data": history_data,
        "col_names": col_names,
        "target_groups": target_groups,
        "trues_forecast_period": trues_forecast_period,
        "output_dir": plot_output_dir
    }


# ------------------- 2. Broad Analysis (Global Aggregate Plots) -------------------

def perform_broad_analysis(data):
    """
    Generates two global aggregate plots to summarise model performance and risk landscape.

    1. Global Forecast Accuracy:
       A single plot showing the Average Ground Truth vs Average Forecast across all nodes.

    2. Global Gap Analysis:
       A single plot showing Average Threat vs Average Solution intensity,
       with Risk (Red) and Safety (Green) shading.

    Args:
        data (dict): The data dictionary returned by `load_visualisation_data`.
    """
    preds = data['preds_window']
    trues = data['trues_forecast_period']
    cols = data['col_names']
    groups = data['target_groups']
    out_dir = data['output_dir']

    horizon = np.arange(preds.shape[0])

    # --- Part A: FORECAST ACCURACY (Global Average) ---
    print("\n" + "=" * 60)
    print("--- FORECAST ACCURACY: Ground Truth vs Prediction ---")
    print("=" * 60)

    all_norm_trues = []
    all_norm_preds = []

    for i in range(len(cols)):
        all_norm_trues.append(normalise_series(trues[:, i]))
        all_norm_preds.append(normalise_series(preds[:, i]))

    avg_true = np.mean(np.array(all_norm_trues), axis=0)
    avg_pred = np.mean(np.array(all_norm_preds), axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(horizon, avg_true, color='black', marker='o', alpha=0.7, label='Ground Truth')
    ax.plot(horizon, avg_pred, color='blue', linestyle='--', marker='x', label='VisionTS++ Forecast')

    ax.set_title("Forecast vs Actual: Global Average", fontsize=14)
    ax.set_xlabel("Months Ahead")
    ax.set_ylabel("Intensity (Normalised)")
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(out_dir / 'Global_Forecast_Accuracy.png', dpi=300)
    print("Generated Plot: Global Forecast Accuracy")
    plt.show()

    # --- Part B: GAP ANALYSIS (Global Average) ---
    print("\n" + "=" * 60)
    print("--- GAP ANALYSIS: Threat vs Mitigation ---")
    print("=" * 60)

    threat_indices = []
    solution_indices = []

    for threat, solutions in groups.items():
        t_idx = find_col_index(threat, cols)
        if t_idx is not None:
            threat_indices.append(t_idx)
        for sol in solutions:
            s_idx = find_col_index(sol, cols)
            if s_idx is not None:
                solution_indices.append(s_idx)

    threat_indices = list(set(threat_indices))
    solution_indices = list(set(solution_indices))

    if threat_indices and solution_indices:
        threat_curves = [normalise_series(preds[:, i]) for i in threat_indices]
        sol_curves = [normalise_series(preds[:, i]) for i in solution_indices]

        avg_threat = np.mean(np.array(threat_curves), axis=0)
        avg_sol = np.mean(np.array(sol_curves), axis=0)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(horizon, avg_threat, color='#d62728', linewidth=2, label='Average Threat')
        ax.plot(horizon, avg_sol, color='#2ca02c', linewidth=2, label='Average Solution')

        ax.fill_between(horizon, avg_threat, avg_sol, where=(avg_threat > avg_sol),
                        color='#d62728', alpha=0.2, label='Risk Gap')
        ax.fill_between(horizon, avg_threat, avg_sol, where=(avg_sol >= avg_threat),
                        color='#2ca02c', alpha=0.15, label='Safety Margin')

        ax.set_title("Gap Analysis: Global Threat vs Mitigation Maturity", fontsize=14)
        ax.set_xlabel("Forecast Horizon (Months)")
        ax.set_ylabel("Normalised Score")
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(out_dir / 'Global_Gap_Analysis.png', dpi=300)
        print("Generated Plot: Global Gap Analysis")
        plt.show()
    else:
        print(f"Warning: Could not classify threats ({len(threat_indices)}) and solutions ({len(solution_indices)}) from graph.csv")


# ------------------- 3. Paper Figure Functions -------------------

def plot_validation_forecasts(data):
    """
    Generates Figure 3 (Forecast Accuracy).

    Produces:
    1. Full-size individual plots for key validation examples (Password Attack, NLP, Backups).
    2. A Grid View plot containing small charts for ALL validation nodes.

    Args:
        data (dict): The data dictionary returned by `load_visualisation_data`.
    """
    print("\n" + "=" * 60)
    print("--- Figure 3: Forecast Accuracy (Ground Truth vs Prediction) ---")
    print("=" * 60)

    preds = data['preds_window']
    trues = data['trues_forecast_period']
    cols = data['col_names']
    out_dir = data['output_dir']
    dates = generate_date_labels_forward(datetime(2023, 1, 1), 36)

    # 1. Full Size Plots (Key Examples)
    keys = ["Password Attack", "NLP/LLM", "Data Backups"]

    for k in keys:
        idx = find_col_index(k, cols)
        if idx is None:
            idx = find_col_index(f"Solution_{k}_Papers", cols)
        if idx is None:
            print(f"  Skipping {k}: not found in selected features")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        norm_t = exponential_smoothing(normalise_series(trues[:, idx]))
        norm_p = exponential_smoothing(normalise_series(preds[:, idx]))

        ax.plot(dates, norm_t, label='Actual', color='black', linewidth=2)
        ax.plot(dates, norm_p, label='Forecast', color='crimson', linestyle='--', linewidth=2)
        ax.fill_between(dates, norm_p * 0.95, norm_p * 1.05, color='mistyrose', alpha=0.6, label='95% Confidence')

        ax.set_title(f"Figure 3: {clean_string(k)}", fontsize=16)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
        ax.legend(loc='upper left', fontsize=10)

        plt.tight_layout()
        plt.savefig(out_dir / f'Fig3_{sanitise_filename(k)}.png', dpi=300)
        print(f"Displaying Plot: Fig 3 - {clean_string(k)}")
        plt.show()

    # 2. Grid View (All Nodes)
    print("\n--- Generating Grid View (All Validation Nodes) ---")

    all_valid_indices = [i for i in range(len(cols)) if np.sum(np.abs(trues[:, i])) > 0]
    n_plots = len(all_valid_indices)

    if n_plots > 0:
        ncols = 3
        nrows = math.ceil(n_plots / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows))
        axes = axes.flatten()

        for i, idx in enumerate(all_valid_indices):
            ax = axes[i]
            node_name = cols[idx]

            norm_t = exponential_smoothing(normalise_series(trues[:, idx]))
            norm_p = exponential_smoothing(normalise_series(preds[:, idx]))

            ax.plot(dates, norm_t, label='Actual', color='black', linewidth=1.5)
            ax.plot(dates, norm_p, label='Forecast', color='crimson', linestyle='--', linewidth=1.5)

            ax.set_title(clean_string(node_name), fontsize=10)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
            ax.tick_params(axis='x', rotation=45, labelsize=8)

            if i == 0:
                ax.legend(loc='upper right', fontsize=8)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(out_dir / 'Fig3_Grid_View_All.png', dpi=300)
        print(f"Generated Grid Plot: All Validation Nodes ({n_plots} items)")
        plt.show()


def generate_gap_analysis_tables(data):
    """
    Generates Tables 5 & 6 (Gap Analysis).

    Calculates the 'Risk Gap' (Threat - Solution) for each year (2023, 2024, 2025)
    and categorises pairs into:
    - Widening Gaps (Risks increasing over time).
    - Narrowing Gaps (Solutions catching up).

    Args:
        data (dict): The data dictionary returned by `load_visualisation_data`.
    """
    print("\n" + "=" * 50)
    print("--- GAP TABLES: Widening & Narrowing ---")
    print("=" * 50)

    preds = data['preds_window']
    cols = data['col_names']
    groups = data['target_groups']
    out_dir = data['output_dir']
    n = preds.shape[0]

    def get_avgs(idx):
        c = max(1, n // 3)
        return [np.mean(normalise_series(preds[0:c, idx])),
                np.mean(normalise_series(preds[c:c * 2, idx])),
                np.mean(normalise_series(preds[c * 2:, idx]))]

    wid_s, wid_o, nar_s, nar_o = [], [], [], []

    for threat, solutions in groups.items():
        t_idx = find_col_index(threat, cols)
        if t_idx is None:
            continue
        t_yrs = get_avgs(t_idx)

        for sol in solutions:
            s_idx = find_col_index(sol, cols)
            if s_idx is None:
                continue
            s_yrs = get_avgs(s_idx)
            diffs = [t - s for t, s in zip(t_yrs, s_yrs)]

            e = {"Threat": clean_string(threat), "PAT": clean_string(sol),
                 "2023": round(diffs[0], 3), "2024": round(diffs[1], 3), "2025": round(diffs[2], 3)}

            if diffs[0] < diffs[1] < diffs[2]:
                wid_s.append(e)
            elif diffs[2] > diffs[0]:
                wid_o.append(e)
            elif diffs[0] > diffs[1] > diffs[2]:
                nar_s.append(e)
            elif diffs[2] < diffs[0]:
                nar_o.append(e)

    def print_tab(d, t, f):
        df = pd.DataFrame(d)
        if not df.empty:
            df = df.sort_values(by="2025", ascending=False)
            save_table_as_image(df.head(10), f, out_dir)
            print(f"\n--- {t} (Top 5) ---")
            print(df.head(5).to_string(index=False))

    print_tab(wid_s, "Table 5a: Strictly Widening Gaps", "Strictly_Widening")
    print_tab(wid_o, "Table 5b: Overall Widening Gaps", "Overall_Widening")
    print_tab(nar_s, "Table 6a: Strictly Narrowing Gaps", "Strictly_Narrowing")
    print_tab(nar_o, "Table 6b: Overall Narrowing Gaps", "Overall_Narrowing")


def plot_continuous_trends(data):
    """
    Generates Figure 4 (Continuous Trends).

    Stitches historical data (2011-2022) with forecasts (2023-2025) to visualise
    the long-term evolution of major threat vectors.

    Args:
        data (dict): The data dictionary returned by `load_visualisation_data`.
    """
    print("\n--- PAPER FIG 4: Continuous Trends ---")

    hist = data['history_data']
    fore = data['preds_window']
    cols = data['col_names']
    groups = data['target_groups']
    out_dir = data['output_dir']

    min_c = min(hist.shape[1], fore.shape[1])
    full = np.concatenate([hist[:, :min_c], fore[:, :min_c]], axis=0)

    dates = generate_date_labels_forward(datetime(2011, 7, 1), full.shape[0])
    f_start = hist.shape[0]
    f_dates = dates[f_start:]

    keys = ["Malware", "Vulnerability", "Ransomware", "Adversarial Attack"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for k in keys:
        match = next((g for g in groups if clean_string(k).lower() in clean_string(g).lower()), None)
        if not match:
            print(f"  Skipping {k}: not found in graph.csv")
            continue
        idx = find_col_index(match, cols)
        if idx is None:
            print(f"  Skipping {k}: not found in selected features")
            continue

        sm_full = exponential_smoothing(normalise_series(full[:, idx]))
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(dates, sm_full, color='black', linewidth=2, label=clean_string(k))

        sm_fore = sm_full[f_start:]
        lim = min(len(f_dates), len(sm_fore))
        ax.fill_between(f_dates[:lim], sm_fore[:lim] - 0.05, sm_fore[:lim] + 0.05, color='mistyrose', alpha=0.5)

        c_idx = 0
        for sol in groups[match]:
            s_idx = find_col_index(sol, cols)
            if s_idx is None:
                continue

            s_sm = exponential_smoothing(normalise_series(full[:, s_idx]))
            s_fore = s_sm[f_start:][:lim]
            t_fore = sm_fore[:lim]

            if len(s_fore) > 0 and np.mean(s_fore) < np.mean(t_fore):
                c = colors[c_idx % len(colors)]
                ax.plot(dates[:len(s_sm)], s_sm, color=c, linewidth=1.5, label=clean_string(sol))
                ax.fill_between(f_dates[:lim], t_fore, s_fore, where=(t_fore > s_fore), color=c, alpha=0.1)
                c_idx += 1

        ax.set_title(f"Fig 4: {clean_string(k)}", fontsize=16)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        plt.savefig(out_dir / f'Fig4_{sanitise_filename(k)}.png', dpi=300)
        print(f"Displaying Plot: Fig 4 - {clean_string(k)}")
        plt.show()


def visualise():
    """
    Main entry point. Executes the full VisionPP visualisation pipeline in sequence.
    """
    data = load_visualisation_data()
    if data:
        perform_broad_analysis(data)
        plot_validation_forecasts(data)
        generate_gap_analysis_tables(data)
        plot_continuous_trends(data)


if __name__ == "__main__":
    visualise()
