import os
import numpy as np
import matplotlib.pyplot as plt
from Utils.Metrics import calculate_gap

# Configuration
RESULTS_DIR = 'Results'
PREDS_FILE = os.path.join(RESULTS_DIR, 'predictions.npy')
TRUES_FILE = os.path.join(RESULTS_DIR, 'ground_truth.npy')

def visualise():
    # 1. Load Data
    print(f"--- Loading Data from {RESULTS_DIR} ---")
    if not os.path.exists(PREDS_FILE) or not os.path.exists(TRUES_FILE):
        raise FileNotFoundError("Prediction files not found. Run Evaluate_Graph.py first.")
        
    preds = np.load(PREDS_FILE)
    trues = np.load(TRUES_FILE)
    
    print(f"Data Shape: {preds.shape} (Samples, Time, Nodes)")

    # 2. Plot A: Forecast vs Ground Truth
    # Visualising the first sample, first node (e.g., DDoS)
    sample_idx = 0
    node_idx = 0
    
    plt.figure(figsize=(12, 6))
    plt.plot(trues[sample_idx, :, node_idx], label='Ground Truth', marker='o', color='black', alpha=0.7)
    plt.plot(preds[sample_idx, :, node_idx], label='PDFormer Forecast', marker='x', linestyle='--', color='blue')
    
    plt.title(f"Forecast vs Actual (Sample {sample_idx}, Node {node_idx})")
    plt.xlabel("Months Ahead")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path_a = os.path.join(RESULTS_DIR, 'forecast_vs_actual.png')
    plt.savefig(output_path_a, dpi=300)
    plt.close()
    print(f"Saved: {output_path_a}")

    # 3. Plot B: Gap Analysis (Threat vs Mitigation)
    # Define which columns correspond to the Threat and its Mitigation
    # CHANGE THESE INDICES to match your dataset columns
    threat_idx = 0      
    mitigation_idx = 1  
    
    threat_seq = preds[sample_idx, :, threat_idx]
    mitigation_seq = preds[sample_idx, :, mitigation_idx]
    gap_seq = calculate_gap(threat_seq, mitigation_seq)
    
    plt.figure(figsize=(12, 6))
    
    # Plot the main lines
    plt.plot(threat_seq, color='#d62728', linewidth=2, label='Projected Threat') # Red
    plt.plot(mitigation_seq, color='#2ca02c', linewidth=2, label='Mitigation Capacity') # Green
    
    # Fill the 'Risk Gap' (Where Threat > Mitigation)
    plt.fill_between(range(len(threat_seq)), threat_seq, mitigation_seq, 
                     where=(threat_seq > mitigation_seq), 
                     color='#d62728', alpha=0.2, hatch='//', label='Risk Gap')
    
    # Fill the 'Safety Margin' (Where Mitigation > Threat)
    plt.fill_between(range(len(threat_seq)), threat_seq, mitigation_seq, 
                     where=(threat_seq <= mitigation_seq), 
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