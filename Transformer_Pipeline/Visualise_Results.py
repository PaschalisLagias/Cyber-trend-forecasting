import os
import json
import numpy as np
import matplotlib.pyplot as plt
# from Utils.Metrics import calculate_gap

# Configuration
RESULTS_DIR = 'Results'
PREDS_FILE = os.path.join(RESULTS_DIR, 'predictions.npy')
TRUES_FILE = os.path.join(RESULTS_DIR, 'ground_truth.npy')
COLS_FILE = os.path.join(RESULTS_DIR, 'column_names.json')

# Dataset columns for mappings
def get_col_index(name, all_cols):
    """Helper to find index of a specific column name."""
    try:
        return all_cols.index(name)
    except ValueError:
        print(f"WARNING: Column '{name}' not found in dataset!")
        print("Available columns snippet:", all_cols[:5], "...")
        raise

def visualise():
    # 1. Load Data
    print(f"--- Loading Data from {RESULTS_DIR} ---")
    if not os.path.exists(PREDS_FILE) or not os.path.exists(TRUES_FILE):
        raise FileNotFoundError("Prediction files not found. Run Evaluate_Graph.py first.")
        
    preds = np.load(PREDS_FILE)
    trues = np.load(TRUES_FILE)
    
    with open(COLS_FILE, 'r') as f:
        col_names = json.load(f)
    
    print(f"Data Shape: {preds.shape} (Samples, Time, Nodes)")

    # 2. Plot A: Forecast vs Ground Truth
    # Based on your CSV, we pair DDoS (Attack) with IDS/IPS (Defense)
    THREAT_NAME = "DDoS-ALL"
    MITIGATION_NAME = "Solution_IDS/IPS_Papers"
    
    # Sample index to plot (0 = the first window in the test set)
    sample_idx = 0

    # Find indices dynamically
    node_idx = get_col_index(THREAT_NAME, col_names)
    mitigation_idx = get_col_index(MITIGATION_NAME, col_names)
    
    # 3. Plot A: Forecast vs Ground Truth
    plt.figure(figsize=(12, 6))
    plt.plot(trues[sample_idx, :, node_idx], label='Ground Truth', marker='o', color='black', alpha=0.7)
    plt.plot(preds[sample_idx, :, node_idx], label='PDFormer Forecast', marker='x', linestyle='--', color='blue')
    
    plt.title(f"Forecast vs Actual: {THREAT_NAME}")
    plt.xlabel("Months Ahead")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path_a = os.path.join(RESULTS_DIR, 'forecast_vs_actual.png')
    plt.savefig(output_path_a, dpi=300)
    plt.close()
    print(f"Saved: {output_path_a}")

    # 4. Plot B: Gap Analysis (Threat vs Mitigation)
    threat_seq = preds[sample_idx, :, node_idx]
    mit_seq = preds[sample_idx, :, mitigation_idx]
    
    
    plt.figure(figsize=(12, 6))
    
    # Plot the main lines
    plt.plot(threat_seq, color='#d62728', linewidth=2, label=f'Threat: {THREAT_NAME}') # red
    plt.plot(mit_seq, color='#2ca02c', linewidth=2, label=f'Def: {MITIGATION_NAME}') # green
    
    # Fill the 'Risk Gap' (Where Threat > Mitigation)
    plt.fill_between(range(len(threat_seq)), threat_seq, mit_seq, 
                     where=(threat_seq > mit_seq), 
                     color='#d62728', alpha=0.2, hatch='//', label='Risk Gap')
    
    # Fill the 'Safety Margin' (Where Mitigation > Threat)
    plt.fill_between(range(len(threat_seq)), threat_seq, mit_seq, 
                     where=(threat_seq <= mit_seq), 
                     color='#2ca02c', alpha=0.1, label='Safety Margin')

    plt.title("Gap Analysis: Threat Intensity vs Mitigation Maturity")
    plt.xlabel("Forecast Horizon (Months)")
    plt.ylabel("Normalized Score")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path_b = os.path.join(RESULTS_DIR, 'gap_analysis.png')
    plt.savefig(output_path_b, dpi=300)
    plt.close()
    print(f"Saved: {output_path_b}")

if __name__ == "__main__":
    visualise()