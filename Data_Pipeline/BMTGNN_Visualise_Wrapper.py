"""
BMTGNN Visualise Wrapper
========================
Imports plotting utilities from Transformer_Pipeline to process B-MTGNN Mark 3.3 
SARIMAX prediction arrays and output figures to an isolated directory.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Path setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

# Append Transformer_Pipeline directory to import plotting logic
transformer_dir = project_root / "Transformer_Pipeline"
if str(transformer_dir) not in sys.path:
    sys.path.append(str(transformer_dir))

try:
    from Visualise_Results import (
        plot_validation_forecasts,
        generate_gap_analysis_tables,
        plot_continuous_trends,
        perform_broad_analysis,
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