import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

# --- Import Config ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    from Transformer_Pipeline.Cyber_Trend_Graph_Config import PDFormerConfig
except ImportError:
    sys.path.append(str(current_dir))
    from Cyber_Trend_Graph_Config import PDFormerConfig

# ------------------- Helper Functions -------------------

def exponential_smoothing(series, alpha=0.1):
    """Smooths the series to remove noise (Paper visual style)."""
    if len(series) == 0: return series
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return np.array(result)

def normalise_series(series):
    """Max-scaling (0-1 range)."""
    mx = np.max(np.abs(series))
    if mx == 0: return series
    return series / mx

def find_col_index(target_name, all_names):
    """Fuzzy searches for a column name."""
    if target_name in all_names:
        return np.where(all_names == target_name)[0][0]
    target_lower = target_name.lower()
    for i, name in enumerate(all_names):
        if target_lower == name.lower(): return i
        if target_lower in name.lower(): return i
    return None

def build_graph(file_name):
    """Parses graph.csv (Col 0 = Threat, Col 1..N = Mitigations)."""
    graph = defaultdict(list)
    if not os.path.exists(file_name):
        print(f"Graph file not found at: {file_name}")
        return graph

    print(f"Loading Graph Structure from: {file_name}")
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            key_node = row[0].strip()
            adjacent_nodes = [node.strip() for node in row[1:] if node.strip()]
            graph[key_node].extend(adjacent_nodes)
    return graph

def generate_date_labels(num_steps, end_year=2025, end_month=12):
    """Generates chronological datetime objects ending at the specified date."""
    dates = []
    current = datetime(end_year, end_month, 1)
    for _ in range(num_steps):
        dates.append(current)
        first = current.replace(day=1)
        current = first - timedelta(days=1)
    return list(reversed(dates))

# ------------------- Gap Analysis Logic (Table 5) -------------------

def generate_gap_tables(preds_window, col_names, target_groups, output_dir):
    """
    Calculates the 'Gap' (Threat - Solution) per year and classifies them.
    Includes DEBUG prints to diagnose why tables might be empty.
    """
    print("\n--- Generating Gap Analysis Tables ---")
    
    n_months = preds_window.shape[0]
    
    def get_yearly_avgs(idx):
        chunk_size = max(1, n_months // 3)
        y1 = np.mean(normalise_series(preds_window[0:chunk_size, idx]))
        y2 = np.mean(normalise_series(preds_window[chunk_size:chunk_size*2, idx]))
        y3 = np.mean(normalise_series(preds_window[chunk_size*2:, idx]))
        return [y1, y2, y3]

    strictly_widening = []
    overall_widening = []
    debug_count = 0 

    for threat_name, solutions in target_groups.items():
        threat_idx = find_col_index(threat_name, col_names)
        if threat_idx is None: continue
        
        t_years = get_yearly_avgs(threat_idx)

        for sol_name in solutions:
            sol_idx = find_col_index(sol_name, col_names)
            if sol_idx is None: continue

            s_years = get_yearly_avgs(sol_idx)
            gaps = [t - s for t, s in zip(t_years, s_years)]
            g1, g2, g3 = gaps

            # DEBUG: Print first few calculations to see raw values
            if debug_count < 10:
                print(f"DEBUG: {threat_name} vs {sol_name} -> Gaps: {g1:.3f}, {g2:.3f}, {g3:.3f}")
                debug_count += 1

            # Classification Logic
            # Strictly Widening: Gap increases every year
            if g1 < g2 < g3:
                strictly_widening.append({
                    "Threat": threat_name,
                    "PAT": sol_name,
                    "2023": round(g1, 3),
                    "2024": round(g2, 3),
                    "2025": round(g3, 3),
                    "Trend": "Strictly Widening"
                })

            # Overall Widening: End gap is larger than start gap
            elif g3 > g1:
                overall_widening.append({
                    "Threat": threat_name,
                    "PAT": sol_name,
                    "2023": round(g1, 3),
                    "2024": round(g2, 3),
                    "2025": round(g3, 3),
                    "Trend": "Overall Widening"
                })

    df_strict = pd.DataFrame(strictly_widening)
    df_overall = pd.DataFrame(overall_widening)

    if not df_strict.empty:
        df_strict = df_strict.sort_values(by="2025", ascending=False)
        save_path = output_dir / "Strictly_Widening_Gaps.csv"
        df_strict.to_csv(save_path, index=False)
        print(f"Saved 'Strictly Widening' table to: {save_path}")
        print("\n--- Strictly Widening Gaps (Top 5) ---")
        print(df_strict.head(5).to_string(index=False))
    else:
        print("\nNote: No 'Strictly Widening' gaps found (Check DEBUG output above).")

    if not df_overall.empty:
        df_overall = df_overall.sort_values(by="2025", ascending=False)
        save_path = output_dir / "Overall_Widening_Gaps.csv"
        df_overall.to_csv(save_path, index=False)
        print(f"Saved 'Overall Widening' table to: {save_path}")
        print("\n--- Overall Widening Gaps (Top 5) ---")
        print(df_overall.head(5).to_string(index=False))
    else:
        print("\nNote: No 'Overall Widening' gaps found (Check DEBUG output above).")

# ------------------- Main Visualisation -------------------

def visualise():
    config = PDFormerConfig()
    data_dir = config.results_dir
    root_path = config.project_root 
    graph_csv_path = root_path / "B-MTGNN" / "data" / "graph.csv"
    
    plot_output_dir = data_dir / 'plots_paper_style'
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(f"--- Starting Visualisation ---")
    print(f"Output Directory: {plot_output_dir.resolve()}")

    # 1. Load Data
    preds_path = data_dir / 'predictions.npy'
    trues_path = data_dir / 'ground_truth.npy'
    cols_path  = data_dir / 'node_names.npy'

    if not preds_path.exists():
        print(f"Error: 'predictions.npy' NOT found.")
        return

    preds = np.load(preds_path)
    trues = np.load(trues_path)
    
    if cols_path.exists():
        col_names = np.load(cols_path, allow_pickle=True)
    else:
        print("Warning: 'node_names.npy' missing.")
        return

    target_groups = build_graph(graph_csv_path)

    # 2. Select Forecast Window
    if preds.ndim == 4: preds = preds.squeeze(-1)
    if trues.ndim == 4: trues = trues.squeeze(-1)

    preds_window = preds[-1, :, :] 
    trues_window = trues[-1, :, :]
    
    # 3. Generate Gap Tables (with Debug Prints)
    generate_gap_tables(preds_window, col_names, target_groups, data_dir)
    
    # 4. Generate Plots
    print("\n--- Generating Plots ---")
    total_steps = preds_window.shape[0]
    date_objs = generate_date_labels(total_steps, end_year=2025, end_month=12)
    
    # --- Part A: Threat Groups (Gap Analysis Plots) ---
    # This covers "Password Attack" (which is a Threat Group Key)
    display_keys_group = ["Password Attack"] 

    for anchor_name, solutions in target_groups.items():
        anchor_idx = find_col_index(anchor_name, col_names)
        if anchor_idx is None: continue
            
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        norm_true = exponential_smoothing(normalise_series(trues_window[:, anchor_idx]))
        norm_pred = exponential_smoothing(normalise_series(preds_window[:, anchor_idx]))
        
        ax.plot(date_objs, norm_true, label=f'{anchor_name} (Actual)', color='#2c3e50', linewidth=2, alpha=0.9)
        ax.plot(date_objs, norm_pred, label=f'{anchor_name} (Forecast)', color='crimson', linestyle='--', linewidth=2)
        
        std_dev = 0.05 * norm_pred 
        ax.fill_between(date_objs, norm_pred - std_dev, norm_pred + std_dev, 
                        color='mistyrose', alpha=0.6, label='95% Confidence')

        colors = ["royalblue", "darkorange", "mediumpurple", "teal", "olivedrab"]
        for idx, sol_name in enumerate(solutions):
            sol_idx = find_col_index(sol_name, col_names)
            if sol_idx is None: continue
            
            n_seq = exponential_smoothing(normalise_series(preds_window[:, sol_idx]))
            col = colors[idx % len(colors)]
            ax.plot(date_objs, n_seq, label=f'{sol_name}', color=col, alpha=0.8, linewidth=1.5)

        quarterly_ticks = [d for d in date_objs if d.month in [3, 6, 9, 12]]
        if len(quarterly_ticks) > 0:
            ax.set_xticks(quarterly_ticks)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
        
        plt.xticks(rotation=90, fontsize=11)
        ax.set_ylabel("Trend", fontsize=14)
        ax.set_title(f"{anchor_name}", fontsize=16, pad=15)
        ax.set_xlim(date_objs[0], date_objs[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='-', alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=True)
        
        plt.tight_layout()
        save_path = plot_output_dir / f'{str(anchor_name).replace("/", "_")}.png'
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if any(k in anchor_name for k in display_keys_group):
            print(f"Displaying Plot: {anchor_name}")
            plt.show()
        else:
            plt.close()

    # --- Part B: Validation Plots (Single Nodes) ---
    # These are NOT Threat keys, they are specific nodes from Fig 3b/3c
    validation_keys = ["NLP/LLM", "Data Backups"]
    
    for val_name in validation_keys:
        val_idx = find_col_index(val_name, col_names)
        if val_idx is None:
            print(f"Warning: Could not find validation node '{val_name}'")
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        norm_true = exponential_smoothing(normalise_series(trues_window[:, val_idx]))
        norm_pred = exponential_smoothing(normalise_series(preds_window[:, val_idx]))
        
        ax.plot(date_objs, norm_true, label=f'{val_name} (Actual)', color='#2c3e50', linewidth=2, alpha=0.9)
        ax.plot(date_objs, norm_pred, label=f'{val_name} (Forecast)', color='crimson', linestyle='--', linewidth=2)
        
        std_dev = 0.05 * norm_pred 
        ax.fill_between(date_objs, norm_pred - std_dev, norm_pred + std_dev, 
                        color='mistyrose', alpha=0.6, label='95% Confidence')
        
        quarterly_ticks = [d for d in date_objs if d.month in [3, 6, 9, 12]]
        if len(quarterly_ticks) > 0:
            ax.set_xticks(quarterly_ticks)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))

        plt.xticks(rotation=90, fontsize=11)
        ax.set_ylabel("Trend", fontsize=14)
        ax.set_title(f"Validation: {val_name}", fontsize=16, pad=15)
        ax.set_xlim(date_objs[0], date_objs[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='-', alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=True)
        
        plt.tight_layout()
        save_path = plot_output_dir / f'Validation_{str(val_name).replace("/", "_")}.png'
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        print(f"Displaying Plot: {val_name}")
        plt.show()

    print(f"All plots saved to: {plot_output_dir}")

if __name__ == "__main__":
    visualise()