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
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List
import sys

# --- Path Configuration ---
# from config.Paths import *
from config.Paths import (
    BMTGNN_WORKING_TXT, BMTGNN_DIR, ROOT_DIR, BMTGNN_SM_DATA_G_CSV,
    BMTGNN_PREDICTIONS, BMTGNN_CONFIDENCE, BMTGNN_HISTORY, BMTGNN_NAMES,
    BMTGNN_EXPLAINABILITY_DIR
    )

# Ensure Python can find the Config folder
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Inherit paths directly from Config
DATA_FILE = BMTGNN_WORKING_TXT 
MODEL_FILE = BMTGNN_DIR / 'model' / 'Bayesian' / 'o_model.pt'
MAPPING_FILE = ROOT_DIR / 'Data_Pipeline' / 'column_mapping.csv'
LEGACY_HEADER_FILE = BMTGNN_SM_DATA_G_CSV

# # Legacy paths required for structural mapping (Updated to point to Zaid's original file)
# # --- Mark 2 Legacy Path (Maintained for Reproducibility) ---
# LEGACY_NODES_FILE = PROJECT_ROOT / 'B-MTGNN' / 'data' / 'data.csv'

# --- The Fix: Inject B-MTGNN into the Python Path ---
B_MTGNN_DIR = PROJECT_ROOT / 'B-MTGNN'
if str(B_MTGNN_DIR) not in sys.path:
    sys.path.append(str(B_MTGNN_DIR))
    print(f"Injected {B_MTGNN_DIR} into system path so PyTorch can find 'net.py'")


def consistent_name(name: str) -> str:
    """
    Normalize names of technologies or threats to a consistent format:
    - Removes suffixes/prefixes such as `-ALL`, `Mentions-`, ` ALL`, 
    `Solution_`, and `_Mentions`.
    - Maps any name containing `HIDDEN MARKOV MODEL` to `SHMM`.
    - Preserves `CAPTCHA`, `DNSSEC`, and `RRAM` unchanged.
    - Title-cases mixed-case names while keeping short words such as `of` 
    lowercase.
    - For uppercase names, converts longer words to title case while 
    preserving:
        - Words of three characters or fewer
        - Words containing `/`
        - `MITM` and `SIEM`

    :param name: Input name
    :return: Normalized name
    """
    name = name \
        .replace('-ALL', '') \
        .replace('Mentions-', '') \
        .replace(' ALL', '') \
        .replace('Solution_', '') \
        .replace('_Mentions', '')
    
    # special cases
    if 'HIDDEN MARKOV MODEL' in name:
        return 'SHMM'

    if name in {'CAPTCHA', 'DNSSEC', 'RRAM'}:
        return name

    # e.g., University of london
    if not name.isupper():
        words = name.split(' ')
        result = ''

        for i, word in enumerate(words):
            if len(word) <= 2:  # e.g., "of"
                result += word
            else:
                result += word[0].upper() + word[1:]
            
            if i < len(words) - 1:
                result += ' '
        return result

    words = name.split(' ')
    result = ''
    for i, word in enumerate(words):
        if len(word) <= 3 or '/' in word or word in {'MITM', 'SIEM'}:
            result += word
        else:
            result += word[0] + (word[1:].lower())
        if i < len(words) - 1:
            result += ' '
    return result


def create_columns(legacy_header_path, mapping_csv_path):
    """
    Constructs ordered column names for Mark 3 dataset by translating
    legacy header ordering through the column mapping table.
    """
    
    # 1. Read legacy header to establish exact column order
    legacy_df = pd.read_csv(legacy_header_path, nrows=0)
    legacy_cols = [c for c in legacy_df.columns if 'date' not in c.lower() and 'month' not in c.lower()]
    
    # 2. Read column mapping dictionary
    mapping_df = pd.read_csv(mapping_csv_path)
    mapping_dict = dict(zip(mapping_df['Legacy_Node_Name'], mapping_df['New_Dataset_Column']))
    mapping_dict["Solution_MACHINE LEARNING_Mentions"] = "Solution_ML/DL_Papers"
    
    # 3. Construct ordered list of new column names
    col_name = []
    col_index = {}
    for old_col in legacy_cols:
        new_col = mapping_dict.get(old_col)
        if not new_col:
            raise KeyError(f"No mapping found for legacy node '{old_col}'")
        col_name.append(new_col)
        
    for i, c in enumerate(col_name):
        col_index[c] = i
        
    return col_name, col_index


def save_attention_scores(scores: torch.Tensor, names: List[str]) -> None:
    """
    Save the torch tensor with attention scores.

    :param scores: Torch tensor with attention scores.
    :param names: List of node names.
    """
    BMTGNN_EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'attention_scores': scores.detach().cpu(),
            'names': names,
            'rows': list(range(len(names))),
            'columns': list(range(len(names))),
            'title': 'Attention Scores'

        },
        BMTGNN_EXPLAINABILITY_DIR / 'attention_scores.pt'
    )


def load_attention_scores(file_path: str) -> Dict[str, Any]:
    """
    Load the torch tensor with attention scores.

    :param file_path: The file path where the tensor is stored.
    :return: Dictionary with attention scores and tensor metadata.
    """
    return torch.load(file_path, map_location='cpu', weights_only=True)


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

    # For Mark 3 Execution:
    col_names, _ = create_columns(LEGACY_HEADER_FILE, MAPPING_FILE)
    
    # NOTE: To revert to Mark 2, simply change the above line to:
    # col_names, _ = create_columns(LEGACY_NODES_FILE, is_mark3=False)

    # 3. Normalise Data (Required by the model architecture)
    scale = np.ones(m)
    dat = np.zeros(rawdat.shape)
    for i in range(m):
        max_val = np.max(np.abs(rawdat[:, i]))
        scale[i] = max_val if max_val > 0 else 1.0
        dat[:, i] = rawdat[:, i] / scale[i]

    # FIX: Initialise Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Prepare the final input window (P=10 lookback)
    P = 10 
    X = torch.from_numpy(dat[-P:, :]) 
    X = torch.unsqueeze(X, dim=0)
    X = torch.unsqueeze(X, dim=1)
    # FIX: Push X to the GPU
    X = X.transpose(2, 3).to(torch.float).to(device)

    # 5. Load Model
    print(f"Loading model weights from: {MODEL_FILE.relative_to(PROJECT_ROOT)}")
    with open(MODEL_FILE, 'rb') as f:
        # weights_only=False required for legacy .pt files
        model = torch.load(f, map_location='cpu')
    
    # FIX: Ensure model is explicitly on the GPU
    model.to(device)

    # 6. Bayesian Estimation (Monte Carlo Runs)
    num_runs = 30
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
    # FIX: Use np.sqrt to avoid CPU tensor initialisation crash
    confidence = z * std_dev / np.sqrt(num_runs)

    # Extract attention scores
    model.eval()
    _, attention_scores = model(X, return_attention=True)
    save_attention_scores(attention_scores, col_names)

    # Clean column names
    names = col_names[:]
    names = [consistent_name(c) for c in names]

    # Column name indexes for plotting attention scores
    attacks = list(range(16))
    attacks_papers = list(range(16, 26))
    solutions_papers = list(range(26, 123))

    # Plot attention scores
    model.visualize_attention_scores(
        names, attacks, attacks, 'Attacks_Attacks', 'Attacks', 'Attacks'
    )
    model.visualize_attention_scores(
        names, attacks_papers, attacks, 'Attacks_Papers_Attacks',
        'Attacks', 'Attacks Papers'
    )
    model.visualize_attention_scores(
        names, solutions_papers, attacks, 'PATs_Attacks', 'Attacks', 'PATs'
    )
    model.visualize_attention_scores(
        names, solutions_papers, solutions_papers, 'PATs_PATs', 'PATs', 'PATs'
    )
    model.visualize_attention_scores(
        names, solutions_papers, attacks_papers, 'PATs_Attacks_Papers',
        'Attacks Papers', 'PATs'
    )

    # 8. Inverse Transform (Scale back to real counts)
    # FIX: Pull GPU tensors to CPU and convert to numpy BEFORE multiplying by the numpy scale array
    dat_unscaled = dat * scale
    Y_unscaled = Y.cpu().numpy() * scale
    confidence_unscaled = confidence.cpu().numpy() * scale

    # 9. Format Tensors to NumPy Arrays 
    # Add a dummy batch dimension (1, 36, 123) to match Vision/PDFormer outputs exactly
    preds_array = np.expand_dims(Y_unscaled, axis=0)
    conf_array = np.expand_dims(confidence_unscaled, axis=0)
    hist_array = dat_unscaled
    names_array = np.array(col_names)

# 10. Save to Processed Data
    print("\n--- Saving Arrays for Phase 5 ---")
    np.save(BMTGNN_PREDICTIONS, preds_array)
    np.save(BMTGNN_CONFIDENCE, conf_array)
    np.save(BMTGNN_HISTORY, hist_array)
    np.save(BMTGNN_NAMES, names_array)

    print(f"Saved: {BMTGNN_PREDICTIONS.name} Shape: {preds_array.shape}")
    print(f"Saved: {BMTGNN_CONFIDENCE.name} Shape: {conf_array.shape}")
    print(f"Saved: {BMTGNN_HISTORY.name} Shape: {hist_array.shape}")
    print(f"Saved: {BMTGNN_NAMES.name} Count: {len(names_array)}")
    print("=" * 60)
    print("SUCCESS: Phase 4 complete. Data is ready for final visualisation.")


if __name__ == "__main__":
    extract_forecast()
