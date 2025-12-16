import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import pickle

# Import custom modules
from Cyber_Trend_Graph_Dataset import CyberThreatGraphDataset
from Models.PDFormer_Wrapper import PDFormer_Wrapper
from Utils.Metrics import calculate_all_metrics, calculate_gap
from Cyber_Trend_Graph_Config import PDFormerConfig

# --- Define SCRIPT_DIR ---
# This gets the absolute path of the folder containing this script
if "__file__" in globals():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    # Fallback for interactive Colab cells if __file__ isn't defined
    SCRIPT_DIR = os.getcwd()

# Now we can safely define Project Root relative to this script
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Configuration
MODEL_PATH = os.path.join(SCRIPT_DIR, 'Checkpoints', 'best_model_graph.pth')
RESULTS_DIR = 'Results'

config = PDFormerConfig()

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
    # Pass the 'split' argument as a positional string
    test_dataset = CyberThreatGraphDataset(config.processed_data_dir, 'test')
    
    # Since we loaded a specific split, we wrap it directly in a DataLoader
    # (We do not use .get_data() here because the dataset object IS the test set)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False
    )

    # # --- DEBUG: DATA Check ---
    # print(f"DEBUG: Dataset Size: {len(test_dataset)}")
    # print(f"DEBUG: Loader Batches: {len(test_loader)}")

    # if len(test_dataset) == 0:
    #     print("CRITICAL WARNING: The Test Dataset is EMPTY! Check Preprocessing.py splits.")
    #     return # Stop here to avoid the crash

    # Get static features for the model wrapper
    static_features = test_dataset.get_static_features()
    
    # 3. Load Model
    print("--- Loading Model ---")
    print(f"Looking for weights at: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Weight file not found! Expected at: {MODEL_PATH}")
    
    # Initialise model with same config structure
    model = PDFormer_Wrapper(config, static_features).to(device)
    
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

    # combine the list of batches into a single array
    preds_array = np.concatenate(all_preds, axis=0)
    trues_array = np.concatenate(all_trues, axis=0)        

    # Handle 4D Output (Squeeze the Feature Dimension) ---
    print(f"DEBUG: Raw Prediction Shape: {preds_array.shape}")
    
    if preds_array.ndim == 4:
        # Turn (Samples, Time, Nodes, 1) -> (Samples, Time, Nodes)
        preds_array = preds_array.squeeze(-1)
        trues_array = trues_array.squeeze(-1)
        print("DEBUG: Squeezed 4D output to 3D.")
    
    print(f"Raw Output Shape: {preds_array.shape}")

    # 5. Inverse Scaling (Crucial for Metrics)
    # We must un-scale data to get real counts (e.g., number of attacks)
    print("--- Inverse Scaling Data ---")
    
    # Iterate through samples and time steps to inverse transform
    # Note: Scaler expects (Samples, Features). We flatten time for efficiency.
    N, T, D = preds_array.shape

    # Load the Scaler manually
    scaler_path = os.path.join(config.processed_data_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from {scaler_path}")
    else:
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    # Reshape to (N*T, D) for scaler
    preds_flat = preds_array.reshape(-1, D)
    trues_flat = trues_array.reshape(-1, D)
    
    # Inverse transform
    preds_unscaled = scaler.inverse_transform(preds_flat).reshape(N, T, D)
    trues_unscaled = scaler.inverse_transform(trues_flat).reshape(N, T, D)

    print("Inverse scaling complete.")

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
    
    print("Loading node names...")
    node_names_path = os.path.join(config.processed_data_dir, 'node_names.npy')

    if os.path.exists(node_names_path):
        # OPTION 1: Load from the preprocessed .npy file (Fastest & Best)
        col_names = np.load(node_names_path, allow_pickle=True)
        print(f"Loaded {len(col_names)} node names from .npy file.")
    else:
        # OPTION 2: Fallback to Raw CSV (Corrected Syntax)
        print("Warning: node_names.npy not found. Reading raw CSV headers.")
        # Use dot notation 'config.raw_data_path', not ['data_path']
        df_head = pd.read_csv(config.raw_data_path, nrows=0)
        # Assuming the first column is Date/Index, take the rest
        col_names = df_head.columns[1:] 
    
    # Ensure list format for plotting
    col_names = list(col_names)

    # # Try to fetch column names from dataset, or fallback if not available
    # if hasattr(test_dataset, 'columns'):
    #     col_names = list(test_dataset.columns)
    # elif hasattr(test_dataset, 'data') and hasattr(test_dataset.data, 'columns'):
    #     col_names = list(test_dataset.data.columns)
    # else:
    #     # Fallback: Load the CSV header just to get names if dataset doesn't expose them
    #     df_head = pd.read_csv(config['data_path'], nrows=0)
    #     col_names = list(df_head.columns)[1:] # Skip 'Date' usually
        
    with open(os.path.join(RESULTS_DIR, 'column_names.json'), 'w') as f:
        json.dump(col_names, f)
        
    print(f"Saved metadata to {RESULTS_DIR}/column_names.json")

# ---------------------------------------------------------
    # --- SAVE ARRAYS (Using Config Results Dir) ---
    # ---------------------------------------------------------
    print("\n--- Saving Raw Outputs for Visualization ---")
    
    # Use the new centralized path from config
    save_dir = config.results_dir 
    
    # Define paths
    preds_save_path = save_dir / 'predictions.npy'
    trues_save_path = save_dir / 'ground_truth.npy'
    node_names_save_path = save_dir / 'node_names.npy'

    # Save
    np.save(preds_save_path, preds_unscaled)
    np.save(trues_save_path, trues_unscaled)
    
    # Save node names too so visualization is self-contained
    if 'col_names' in locals():
        np.save(node_names_save_path, col_names)

    print(f"Saved predictions to: {preds_save_path}")
    print(f"Saved ground truth to: {trues_save_path}")
    print("------------------------------------------------")

if __name__ == "__main__":
    evaluate()