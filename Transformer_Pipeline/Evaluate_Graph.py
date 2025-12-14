import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import custom modules
from Cyber_Trend_Graph_Dataset import Cyber_Trend_Graph_Dataset
from Models.PDFormer_Wrapper import PDFormer  # Assuming class name is PDFormer
from Utils.Metrics import calculate_all_metrics, calculate_gap
from Cyber_Trend_Graph_Config import config

# Configuration
MODEL_PATH = 'best_model_graph.pth'   
RESULTS_DIR = 'Results'

def evaluate():
    # 1. Setup Environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    # print(f"--- Loading Configuration from {CONFIG_PATH} ---")
    # with open(CONFIG_PATH, 'r') as f:
    #     config = json.load(f)

    # 2. Load Test Data
    print("--- Loading Test Dataset ---")
    # Force mode='test' to get the validation/test split
    test_dataset = Cyber_Trend_Graph_Dataset(config, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 3. Load Model
    print("--- Loading Model ---")
    # Initialise model with same config structure
    model = PDFormer(config).to(device)
    
    # Load trained weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Successfully loaded weights from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found! Run Train_Graph.py first.")

    model.eval()
    
    # 4. Inference Loop
    all_preds = []
    all_trues = []
    
    print("--- Running Inference ---")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # Forward pass
            output = model(data)
            # Move to CPU and numpy
            all_preds.append(output.cpu().numpy())
            all_trues.append(target.cpu().numpy())

    # Concatenate all batches -> Shape: (Total_Samples, Time_Steps, Nodes)
    preds_array = np.concatenate(all_preds, axis=0)
    trues_array = np.concatenate(all_trues, axis=0)
    
    print(f"Raw Output Shape: {preds_array.shape}")

    # 5. Inverse Scaling (Crucial for Metrics)
    # We must un-scale data to get real counts (e.g., number of attacks)
    print("--- Inverse Scaling Data ---")
    
    # Iterate through samples and time steps to inverse transform
    # Note: Scaler expects (Samples, Features). We flatten time for efficiency.
    N, T, D = preds_array.shape
    
    # Reshape to (N*T, D) for scaler
    preds_flat = preds_array.reshape(-1, D)
    trues_flat = trues_array.reshape(-1, D)
    
    # Inverse transform
    preds_unscaled = test_dataset.scaler.inverse_transform(preds_flat).reshape(N, T, D)
    trues_unscaled = test_dataset.scaler.inverse_transform(trues_flat).reshape(N, T, D)

    # 6. Generate Metrics Table
    print("--- Calculating Metrics per Horizon ---")
    horizons = [3, 6, 12, 24]
    results_list = []

    for h in horizons:
        # Slice data up to horizon h
        p_slice = preds_unscaled[:, :h, :]
        t_slice = trues_unscaled[:, :h, :]
        
        metrics = calculate_all_metrics(p_slice, t_slice)
        metrics['Horizon'] = f"{h} Months"
        results_list.append(metrics)

    # Create DataFrame and Save
    results_df = pd.DataFrame(results_list)
    # Reorder columns
    results_df = results_df[['Horizon', 'RSE', 'RAE', 'MAE', 'CORR']]
    
    csv_path = os.path.join(RESULTS_DIR, 'graph_evaluation_results.csv')
    results_df.to_csv(csv_path, index=False)
    
    print("\nEvaluation Results:")
    print(results_df)
    print(f"Saved results table to {csv_path}")

    # 7. Save Raw Data & Column Names for Visualization
    print("--- Saving Raw Predictions & Metadata ---")
    np.save(os.path.join(RESULTS_DIR, 'predictions.npy'), preds_unscaled)
    np.save(os.path.join(RESULTS_DIR, 'ground_truth.npy'), trues_unscaled)
    
    # Try to fetch column names from dataset, or fallback if not available
    if hasattr(test_dataset, 'columns'):
        col_names = list(test_dataset.columns)
    elif hasattr(test_dataset, 'data') and hasattr(test_dataset.data, 'columns'):
        col_names = list(test_dataset.data.columns)
    else:
        # Fallback: Load the CSV header just to get names if dataset doesn't expose them
        df_head = pd.read_csv(config['data_path'], nrows=0)
        col_names = list(df_head.columns)[1:] # Skip 'Date' usually
        
    with open(os.path.join(RESULTS_DIR, 'column_names.json'), 'w') as f:
        json.dump(col_names, f)
        
    print(f"Saved metadata to {RESULTS_DIR}/column_names.json")

if __name__ == "__main__":
    evaluate()