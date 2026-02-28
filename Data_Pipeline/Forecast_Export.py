"""
Forecast Extraction Pipeline
=====================================

This script handles the B-MTGNN forecasting with the new 2025 dataset.
It loads a trained model and the unsmoothed data, runs 10 Monte Carlo 
passes to approximate Bayesian inference, and calculates the mean forecast 
and 95% confidence intervals.

Note: it bypasses the legacy plotting functions and intercepts the raw 
tensors, exporting them as structured NumPy arrays (.npy) into the 
Processed_Data folder. These arrays are formatted to match the 
shapes expected by the master Visualise_Results.py pipeline.

Usage:
    Run from the B-MTGNN directory:
    $ python forecast_export.py
"""

import os
import sys
import csv
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys


# --- Path Configuration ---
# --- Path Configuration ---
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Input Paths (pointing to the newly processed data and model)
DATA_DIR = PROJECT_ROOT / "Processed_Data" / "B-MTGNN"
DATA_FILE = DATA_DIR / 'sm_data.txt'
MODEL_FILE = DATA_DIR / 'o_model.pt'

# Legacy paths required for structural mapping (Updated to point to Zaid's original file)
LEGACY_NODES_FILE = PROJECT_ROOT / 'B-MTGNN' / 'data' / 'data.csv'

# Output paths
PREDS_SAVE_PATH = DATA_DIR / 'predictions.npy'
CONF_SAVE_PATH = DATA_DIR / 'confidence.npy'
HIST_SAVE_PATH = DATA_DIR / 'history_data.npy'
NAMES_SAVE_PATH = DATA_DIR / 'node_names.npy'

# --- The Fix: Inject B-MTGNN into the Python Path ---
B_MTGNN_DIR = PROJECT_ROOT / 'B-MTGNN'
if str(B_MTGNN_DIR) not in sys.path:
    sys.path.append(str(B_MTGNN_DIR))
    print(f"Injected {B_MTGNN_DIR} into system path so PyTorch can find 'net.py'")

def create_columns(file_name):
    """Extracts column names and index mapping from the legacy CSV."""
    col_name = []
    col_index = {}
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        col_name = [c for c in next(reader)]
        if 'Date' in col_name[0]:
            col_name = col_name[1:]
        for i, c in enumerate(col_name):
            col_index[c] = i
        return col_name, col_index

def extract_forecast():
    """Executes the Bayesian forecast and exports the NumPy arrays."""
    print("=" * 60)
    print("--- Initiating Phase 4: Forecast Extraction ---")
    print("=" * 60)

    # 1. Verify Model Exists
    if not MODEL_FILE.exists():
        print(f"CRITICAL ERROR: Trained model not found at {MODEL_FILE}")
        print("Please ensure Phase 3 (train.py) completed successfully.")
        sys.exit(1)

    # 2. Load Data and Metadata
    print(f"Loading data from: {DATA_FILE.relative_to(PROJECT_ROOT)}")
    try:
        rawdat = np.loadtxt(DATA_FILE, delimiter='\t')
        n, m = rawdat.shape
        print(f"Data shape loaded: {rawdat.shape} (Months x Nodes)")
    except Exception as e:
        print(f"Failed to load sm_data.txt: {e}")
        sys.exit(1)

    col_names, _ = create_columns(LEGACY_NODES_FILE)

    # 3. Normalise Data (Required by the model architecture)
    scale = np.ones(m)
    dat = np.zeros(rawdat.shape)
    for i in range(m):
        max_val = np.max(np.abs(rawdat[:, i]))
        scale[i] = max_val if max_val > 0 else 1.0
        dat[:, i] = rawdat[:, i] / scale[i]

    # 4. Prepare the final input window (P=10 lookback)
    P = 10 
    X = torch.from_numpy(dat[-P:, :]) 
    X = torch.unsqueeze(X, dim=0)
    X = torch.unsqueeze(X, dim=1)
    X = X.transpose(2, 3).to(torch.float)

    # 5. Load Model
    print(f"Loading model weights from: {MODEL_FILE.relative_to(PROJECT_ROOT)}")
    with open(MODEL_FILE, 'rb') as f:
        # weights_only=False required for legacy .pt files
        model = torch.load(f, weights_only=False)

    # 6. Bayesian Estimation (Monte Carlo Runs)
    num_runs = 10
    outputs = []
    print(f"Executing {num_runs} Monte Carlo inference passes...")
    
    # We turn ON dropout during inference to calculate variance
    model.train() 

    for _ in range(num_runs):
        with torch.no_grad():
            output = model(X)  
            y_pred = output[-1, :, :, -1].clone() # Shape: 36 x 123
        outputs.append(y_pred)

    outputs = torch.stack(outputs)

    # 7. Calculate Mean and Confidence Intervals
    print("Calculating Bayesian Mean and 95% Confidence Intervals...")
    Y = torch.mean(outputs, dim=0)
    std_dev = torch.std(outputs, dim=0)
    
    z = 1.96 # Z-score for 95% confidence
    confidence = z * std_dev / torch.sqrt(torch.tensor(num_runs))

    # 8. Inverse Transform (Scale back to real counts)
    # Using the exact same scale array used during preprocessing
    dat_unscaled = dat * scale
    Y_unscaled = Y * scale
    confidence_unscaled = confidence * scale

    # 9. Format Tensors to NumPy Arrays 
    # Add a dummy batch dimension (1, 36, 123) to match Vision/PDFormer outputs exactly
    preds_array = np.expand_dims(Y_unscaled.cpu().numpy(), axis=0)
    conf_array = np.expand_dims(confidence_unscaled.cpu().numpy(), axis=0)
    hist_array = dat_unscaled
    names_array = np.array(col_names)

    # 10. Save to Processed Data
    print("\n--- Saving Arrays for Phase 5 ---")
    np.save(PREDS_SAVE_PATH, preds_array)
    np.save(CONF_SAVE_PATH, conf_array)
    np.save(HIST_SAVE_PATH, hist_array)
    np.save(NAMES_SAVE_PATH, names_array)

    print(f"Saved: {PREDS_SAVE_PATH.name} Shape: {preds_array.shape}")
    print(f"Saved: {CONF_SAVE_PATH.name} Shape: {conf_array.shape}")
    print(f"Saved: {HIST_SAVE_PATH.name} Shape: {hist_array.shape}")
    print(f"Saved: {NAMES_SAVE_PATH.name} Count: {len(names_array)}")
    print("=" * 60)
    print("SUCCESS: Phase 4 complete. Data is ready for final visualisation.")

if __name__ == "__main__":
    extract_forecast()