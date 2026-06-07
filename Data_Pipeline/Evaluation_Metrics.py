"""
Multi-Horizon Evaluation Metrics
================================

This script loads the trained B-MTGNN model and computes forecasting 
error metrics (RSE, RAE, MAE, CORR) across temporal horizons 
(3, 6, 12, and 24 months) using the testing dataset.

Output is saved as a structured CSV.
"""

import os
import sys
import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Path Configuration
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Inject B-MTGNN into sys.path to access util and net
B_MTGNN_DIR = PROJECT_ROOT / 'B-MTGNN'
if str(B_MTGNN_DIR) not in sys.path:
    sys.path.append(str(B_MTGNN_DIR))

from Config.Paths import *
from util import DataLoaderS

# File configurations
DATA_FILE = BMTGNN_SM_DATA_TXT 
MODEL_FILE = B_MTGNN_DIR / 'model' / 'Bayesian' / 'o_model.pt'
OUTPUT_CSV = CURRENT_DIR / 'Results' / 'graph_evaluation_results.csv'


def compute_metrics(predict, y_test):
    """
    Computes RSE, RAE, MAE, and Pearson Correlation.
    Includes 1e-5 epsilon to prevent division-by-zero on sparse tensors.
    """
    predict_tensor = torch.from_numpy(predict)
    y_test_tensor = torch.from_numpy(y_test)

    diff = predict_tensor - y_test_tensor
    sum_absolute_diff = torch.sum(torch.abs(diff))
    sum_squared_diff = torch.sum(diff * diff)

    mean_r = torch.mean(y_test_tensor, dim=0)
    diff_r = y_test_tensor - mean_r
    sum_squared_r = torch.sum(diff_r * diff_r)
    sum_absolute_r = torch.sum(torch.abs(diff_r)) + 1e-5 

    rse = (math.sqrt(sum_squared_diff) / (math.sqrt(sum_squared_r) + 1e-5))
    rae = (sum_absolute_diff / sum_absolute_r).item()
    mae = torch.mean(torch.abs(diff)).item()

    sigma_p = predict.std(axis=0)
    sigma_g = y_test.std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = y_test.mean(axis=0)
    
    correlation = ((predict - mean_p) * (y_test - mean_g)).mean(axis=0) / (sigma_p * sigma_g + 1e-5)
    corr = np.nanmean(correlation)

    return rse, rae, mae, corr


def generate_horizon_metrics():
    """Loads the model and computes metrics across predefined horizons."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = DataLoaderS(
        file_name=str(DATA_FILE), 
        train=0.43, 
        valid=0.30, 
        device=device, 
        horizon=1, 
        window=10, 
        normalize=2, 
        out=36
    )

    with open(MODEL_FILE, 'rb') as f:
        model = torch.load(f, weights_only=False).to(device)
    
    model.eval()
    
    x_test, y_test = data.test[0].to(device), data.test[1].to(device)
    scale = data.scale.expand(y_test.size(0), y_test.size(1), data.m).to(device)

    # Reshape 3D tensor to 4D tensor [Batch, Channels, Nodes, Window]
    x_test = torch.unsqueeze(x_test, dim=1)
    x_test = x_test.transpose(2, 3)

    with torch.no_grad():
        output = model(x_test)
        output = torch.squeeze(output, 3)

    predict_unscaled = (output * scale).cpu().numpy()
    y_test_unscaled = (y_test * scale).cpu().numpy()

    horizons = [3, 6, 12, 24]
    results = []

    for h in horizons:
        idx = h - 1 
        
        pred_h = predict_unscaled[:, idx, :]
        test_h = y_test_unscaled[:, idx, :]
        
        rse, rae, mae, corr = compute_metrics(pred_h, test_h)
        results.append({
            "Horizon": f"{h} Months",
            "RSE": round(rse, 6),
            "RAE": round(rae, 6),
            "MAE": round(mae, 6),
            "CORR": round(corr, 6)
        })

    rse_o, rae_o, mae_o, corr_o = compute_metrics(predict_unscaled, y_test_unscaled)
    results.append({
        "Horizon": "Overall",
        "RSE": round(rse_o, 6),
        "RAE": round(rae_o, 6),
        "MAE": round(mae_o, 6),
        "CORR": round(corr_o, 6)
    })

    df = pd.DataFrame(results)
    print("Evaluation Results:")
    print(df.to_string(index=False))
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved results table to {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    generate_horizon_metrics()