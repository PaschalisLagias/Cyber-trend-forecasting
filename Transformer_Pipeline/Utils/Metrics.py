import numpy as np

def RSE(pred, true):
    """
    Root Relative Squared Error (RSE)
    Standard metric for B-MTGNN comparison.
    
    Formula: sqrt( sum((pred - true)^2) ) / sqrt( sum((true - mean)^2) )
    """
    true_mean = np.mean(true)
    squared_error = np.sum((pred - true) ** 2)
    squared_variance = np.sum((true - true_mean) ** 2)
    
    # Avoid division by zero
    if squared_variance == 0:
        return 0.0
        
    return np.sqrt(squared_error) / np.sqrt(squared_variance)

def RAE(pred, true):
    """
    Relative Absolute Error (RAE)
    Standard metric for B-MTGNN comparison.
    
    Formula: sum( abs(pred - true) ) / sum( abs(true - mean) )
    """
    true_mean = np.mean(true)
    absolute_error = np.sum(np.abs(pred - true))
    absolute_deviation = np.sum(np.abs(true - true_mean))
    
    if absolute_deviation == 0:
        return 0.0
        
    return absolute_error / absolute_deviation

def MAE(pred, true):
    """
    Mean Absolute Error (MAE)
    Standard metric for Transformer performance.
    
    Formula: mean( abs(pred - true) )
    """
    return np.mean(np.abs(pred - true))

def CORR(pred, true):
    """
    Empirical Correlation Coefficient (CORR)
    Crucial for Graph models to prove they capture trends/causality.
    
    Calculates the average correlation across the batch/nodes.
    """
    # Flatten to 1D vectors for simple correlation if shapes match
    # Or calculate per-node correlation and average
    
    # Handling simple vector correlation for evaluation
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    
    # Avoid correlation of constant sequences
    if np.std(pred_flat) == 0 or np.std(true_flat) == 0:
        return 0.0
        
    return np.corrcoef(pred_flat, true_flat)[0, 1]

def calculate_all_metrics(pred, true):
    """
    Helper to run all metrics at once.
    Expected Input Shape: (Batch, Time, Nodes) or (Time, Nodes)
    """
    return {
        "RSE": RSE(pred, true),
        "RAE": RAE(pred, true),
        "MAE": MAE(pred, true),
        "CORR": CORR(pred, true)
    }

def calculate_gap(threat_seq, mitigation_seq):
    """
    Calculates the 'Gap' (Delta) for the visualisations.
    
    Args:
        threat_seq: Array of predicted threat intensity
        mitigation_seq: Array of predicted mitigation maturity
        
    Returns:
        gap_seq: (Threat - Mitigation)
    """
    return threat_seq - mitigation_seq