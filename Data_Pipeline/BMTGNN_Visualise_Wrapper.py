"""
BMTGNN Visualise Wrapper
========================
Imports plotting utilities from Transformer_Pipeline to process B-MTGNN Mark 3.3 
SARIMAX prediction arrays and output figures to an isolated directory.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Path setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

transformer_dir = project_root / "Transformer_Pipeline"
if str(transformer_dir) not in sys.path:
    sys.path.append(str(transformer_dir))

try:
    from Visualise_Results import (
        plot_validation_forecasts,
        generate_gap_analysis_tables,
        save_table_as_image,
        generate_date_labels_forward,
        clean_string,
        sanitise_filename,
        find_col_index,
        normalise_series,
        double_save_figure,
        build_graph,
        load_full_history
    )
    from Cyber_Trend_Graph_Config import PDFormerConfig
except ImportError as e:
    print(f"Failed to import from Transformer Pipeline: {e}")
    sys.exit(1)


def load_bmtgnn_mark3_data(experiment_suffix="Mark3.3_SARIMAX"):
    """
    Loads B-MTGNN prediction and ground truth arrays from Processed_Data/B-MTGNN/
    and constructs the data dictionary required by the visualisation functions.

    Args:
        experiment_suffix (str): The filename suffix identifying the run.

    Returns:
        dict or None: Data dictionary containing prediction arrays and output path.
    """
    print("=" * 60)
    print(f"--- Loading B-MTGNN Data ({experiment_suffix}) for Visualisation ---")
    
    # Target directory for B-MTGNN plots
    plot_output_dir = project_root / "Data_Pipeline" / "Results" / "BMTGNN_Mark3_Plots"
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = project_root / "Processed_Data" / "B-MTGNN"
    
    preds_path = data_dir / f"predictions_{experiment_suffix}.npy"
    cols_path = data_dir / f"node_names_{experiment_suffix}.npy"
    trues_path = data_dir / f"history_data_{experiment_suffix}.npy"
    graph_csv_path = project_root / "B-MTGNN" / "data" / "graph.csv"

    if not preds_path.exists():
        print(f"Error: File '{preds_path.name}' not found in {data_dir.relative_to(project_root)}")
        return None

    print(f"Loading prediction arrays from: {data_dir.relative_to(project_root)}")
    print(f" - Predictions:  {preds_path.name}")
    print(f" - Node Names:   {cols_path.name}")
    print(f" - Ground Truth: {trues_path.name}")

    preds = np.load(preds_path)
    col_names = np.load(cols_path, allow_pickle=True)
    target_groups = build_graph(graph_csv_path)

    if preds.ndim == 4:
        preds = preds.squeeze(-1)
    preds_window = preds[-1, :, :] if preds.ndim == 3 else preds

    trues_period = np.load(trues_path)
    if trues_period.ndim == 4:
        trues_period = trues_period.squeeze(-1)
    trues_forecast_period = trues_period[-1, :, :] if trues_period.ndim == 3 else trues_period

    # Align the ground truth array to match the prediction window length
    pred_len = preds_window.shape[0]
    if trues_forecast_period.shape[0] > pred_len:
        trues_forecast_period = trues_forecast_period[-pred_len:, :]

    dummy_config = PDFormerConfig()
    history_data, success = load_full_history(dummy_config, col_names)
    
    if not success:
        history_data = trues_forecast_period

    print("B-MTGNN Data mapping successful.")
    
    return {
        "preds_window": preds_window,
        "history_data": history_data,
        "col_names": col_names,
        "target_groups": target_groups,
        "trues_forecast_period": trues_forecast_period,
        "output_dir": plot_output_dir
    }

def generate_bmodel_evaluation_metrics(output_dir, experiment_suffix="Mark3.3_SARIMAX"):
    """
    Loads dynamic evaluation CSVs and generates two tables:
    1. Detailed 5-horizon evaluation for the bmodel.
    2. Simplified comparative evaluation (Table 9 style) for bmodel vs VisionTS++.
    
    Outputs PNG images and Markdown strings.
    
    Args:
        output_dir (Path): The target directory for saved images.
        experiment_suffix (str): The tag used to locate the correct bmodel CSV.
    """
    print("\n" + "=" * 60)
    print("--- Generating Evaluation Metrics (Detailed & Comparative) ---")
    print("=" * 60)
    
# Construct absolute paths to the dynamically generated evaluation CSVs
    bmodel_csv_path = current_dir / "Results" / "graph_evaluation_results.csv"
    vision_csv_path = project_root / "Transformer_Pipeline" / "Results" / "v2_2_full" / "visionpp_evaluation_results.csv"
    
    # Validate file existence before proceeding
    if not bmodel_csv_path.exists():
        print(f"Error: bmodel evaluation CSV not found at {bmodel_csv_path}")
        return
    if not vision_csv_path.exists():
        print(f"Error: VisionTS++ evaluation CSV not found at {vision_csv_path}")
        return

    # Load DataFrames
    df_bmodel = pd.read_csv(bmodel_csv_path)
    df_vision = pd.read_csv(vision_csv_path)
    
    # --- Part 1: Detailed 5-Horizon Table (bmodel) ---
    save_table_as_image(df_bmodel, "Table_8_Detailed_BMTGNN", output_dir)
    print("\nDetailed B-MTGNN Markdown Table:")
    print(df_bmodel.to_markdown(index=False, floatfmt=".3f"))
    
    # --- Part 2: Simplified Comparative Table (Table 9 format) ---
    # Extract the 'Overall' row (36-month horizon) for both models
    bmodel_overall = df_bmodel[df_bmodel["Horizon"] == "Overall"].iloc[0]
    vision_overall = df_vision[df_vision["Horizon"] == "Overall"].iloc[0]
    
    # Construct the simplified comparison mapping
    comparative_data = {
        "Model": ["B-MTGNN", "VisionTS++"],
        "RSE": [bmodel_overall["RSE"], vision_overall["RSE"]],
        "RAE": [bmodel_overall["RAE"], vision_overall["RAE"]]
    }
    
    df_comparative = pd.DataFrame(comparative_data)
    
    # Generate visual PNG and terminal Markdown for the comparative table
    save_table_as_image(df_comparative, "Table_9_Comparative_Evaluation", output_dir)
    print("\nComparative Markdown Table (Table 9):")
    print(df_comparative.to_markdown(index=False, floatfmt=".2f"))
    print("\n" + "=" * 60)


def exponential_smoothing_clamped(series, alpha=0.01):
    """
    Applies exponential smoothing to reduce noise. 
    Explicitly clamps values at 0 to prevent illogical negative graph dips.
    
    Args:
        series (np.array): The input time series data.
        alpha (float): The smoothing factor (0 < alpha <= 1).
        
    Returns:
        np.array: The smoothed time series.
    """
    if len(series) == 0:
        return series
        
    result = [max(0, series[0])]
    for n in range(1, len(series)):
        val = alpha * series[n] + (1 - alpha) * result[n - 1]
        result.append(max(0, val))
        
    return np.array(result)


def plot_bmodel_continuous_trends(data, alpha=0.01):
    """
    Generates Figure 4 (Continuous Trends) for the B-MTGNN model.
    
    Stitches historical data (2011-2022) with forecasts (2023-2025). 
    Logic is retained verbatim from Visualise_Results.py, but uses the clamped 
    smoothing function with a parameterised alpha value.
    
    Args:
        data (dict): The data dictionary returned by the loader.
        alpha (float): The smoothing parameter to apply to the plots.
    """
    print(f"\n--- PAPER FIG 4: Continuous Trends (B-MTGNN, alpha={alpha}) ---")
    
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
        match = None
        for g in groups:
            if clean_string(k).lower() in clean_string(g).lower() or clean_string(g).lower() in clean_string(k).lower():
                match = g
                break
                
        if not match: 
            print(f"Warning: Skipping '{k}' plot. Could not find a match in graph.csv")
            continue
            
        idx = find_col_index(match, cols)
        if idx is None: 
            print(f"Warning: Skipping '{k}' plot. Found in graph but missing from node_names.npy")
            continue
        
        # Apply parameterised clamped smoothing
        sm_full = exponential_smoothing_clamped(normalise_series(full[:, idx]), alpha=alpha)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(dates, sm_full, color='black', linewidth=2, label=clean_string(k))
        
        sm_fore = sm_full[f_start:]
        lim = min(len(f_dates), len(sm_fore))
        ax.fill_between(f_dates[:lim], sm_fore[:lim]-0.05, sm_fore[:lim]+0.05, color='mistyrose', alpha=0.5)
        
        c_idx = 0
        for sol in groups[match]:
            s_idx = find_col_index(sol, cols)
            if s_idx is None: 
                continue
            
            # Apply parameterised clamped smoothing to solutions
            s_sm = exponential_smoothing_clamped(normalise_series(full[:, s_idx]), alpha=alpha)
            s_fore = s_sm[f_start:][:lim]
            t_fore = sm_fore[:lim]
            
            if len(s_fore) > 0 and np.mean(s_fore) < np.mean(t_fore):
                c = colors[c_idx % len(colors)]
                ax.plot(dates[:len(s_sm)], s_sm, color=c, linewidth=1.5, label=clean_string(sol))
                ax.fill_between(f_dates[:lim], t_fore, s_fore, where=(t_fore>s_fore), color=c, alpha=0.1)
                c_idx += 1
        
        ax.set_title(f"Fig 4: {clean_string(k)}", fontsize=16)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        file_name = f'Fig4_{sanitise_filename(k)}.png'
        double_save_figure(out_dir, file_name, show_inline=False, fig=fig)


def execute_all_bmtgnn_visualisations():
    """Runs all visualisation functions for the B-MTGNN Mark3.3 SARIMAX run."""
    vis_data = load_bmtgnn_mark3_data("Mark3.3_SARIMAX")
    if vis_data:
        perform_broad_analysis(vis_data)
        plot_validation_forecasts(vis_data)
        generate_gap_analysis_tables(vis_data)
        plot_continuous_trends(vis_data)


if __name__ == "__main__":
    execute_all_bmtgnn_visualisations()