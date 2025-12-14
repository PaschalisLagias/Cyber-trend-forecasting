# Import Standard python library imports
import os
import pickle
import argparse # Added for command-line flag
import sys
from pathlib import Path

# Import Third-party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Imports from base paper 
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Imports for PDFormer transformations (uncomment when once final test complete)
from fastdtw import fastdtw
from tslearn.clustering import TimeSeriesKMeans

# Local imports
from .Load_Data import load_cyber_threat_data

# ---------------------------------------------------------
# PATH SETUP: Allow importing from Transformer_Pipeline
# ---------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from Transformer_Pipeline.Cyber_Trend_Graph_Config import PDFormerConfig

# ------------------- Preprocessing Functions -------------------

def apply_double_exponential_smoothing(df, alpha=0.1, beta=0.1):
    """
    Applies Double Exponential Smoothing (Holt's Linear Trend) to the dataframe.
    
    Args:
        df (pd.DataFrame): Input data (Time x Nodes).
        alpha (float): Smoothing factor for the level (0 < alpha < 1).
        beta (float): Smoothing factor for the trend (0 < beta < 1).
        
    Returns:
        pd.DataFrame: Smoothed data with the same shape and index.
    """
    print(f"\n--- Applying Double Exponential Smoothing (alpha={alpha}, beta={beta}) ---")
    
    df_smoothed = df.copy()
    
    # Apply to each column (node) independently
    # We use a simple loop because statsmodels operates on 1D series
    success_count = 0
    
    for col in df.columns:
        try:
            # Holt's Linear Trend method
            # initialization_method='estimated' is robust for various data shapes
            model = ExponentialSmoothing(
                df[col], 
                trend='add', 
                seasonal=None, 
                initialization_method='estimated'
            )
            fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
            df_smoothed[col] = fit.fittedvalues
            success_count += 1
        except Exception as e:
            print(f"Warning: Could not smooth column '{col}': {e}. Keeping raw data.")
            
    print(f"Smoothed {success_count}/{len(df.columns)} columns.")
    return df_smoothed

def split_data(df_raw, train_split, val_split):
    """
    Splits the raw DataFrame chronologically into training, validation, and test sets.

    Args:
        df_raw (pd.DataFrame): The raw DataFrame.
        train_split (float): The proportion of data for the training set (e.g., 0.7).
        val_split (float): The proportion of data for the validation set (e.g., 0.1).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames for the training, validation, and test sets.
    """
    # --- PROCESS ---
    # Calculate split indices based on TRAIN_SPLIT and VAL_SPLIT proportions.
    # Create train_df, val_df, test_df pandas DataFrames using iloc.
    # Print the date ranges and sizes of each split for verification.
    # --- CODE ---
    print("\n--- Step 1: Splitting Data ---")
    n = len(df_raw)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    train_df = df_raw.iloc[:train_end]
    val_df = df_raw.iloc[train_end:val_end]
    test_df = df_raw.iloc[val_end:]

    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    print("Data splitting complete")
    return train_df, val_df, test_df


def define_graph_structure(train_df, correlation_threshold, output_dir):
    """
    Defines and saves the base graph structure (adjacency matrix) based on feature correlations.

    Args:
        train_df (pd.DataFrame): The training data DataFrame (used exclusively for calculating correlations).
        correlation_threshold (float): The absolute Pearson correlation value above which an edge is created.
        output_dir (str): The directory path to save the adjacency matrix file.

    Returns:
        np.ndarray or None: The calculated adjacency matrix (NumPy array) or None if calculation fails.
    """
    # --- PROCESS ---
    # Calculate the Pearson correlation matrix *only* using train_df.
    # Create the adjacency matrix based on CORRELATION_THRESHOLD.
    # Ensure the adjacency matrix is symmetric and has self-loops (diagonal = 1).
    # Save the adjacency matrix (e.g., as adj_mx.npy in OUTPUT_DIR).
    print("\n--- Step 2: Defining Graph Structure ---")

    # Calculate correlation based only on training data to avoid leakage
    corr_matrix = train_df.corr()
    
    # Create adjacency: 1 if correlation > threshold, else 0
    adj_matrix = (np.abs(corr_matrix) > correlation_threshold).astype(int)
    # Ensure diagonal is 1 (self-loops) - important for GNNs
    np.fill_diagonal(adj_matrix.values, 1)
    
    # Save data
    adj_matrix_path = os.path.join(output_dir, 'adj_mx.npy')
    np.save(adj_matrix_path, adj_matrix.values)
    print(f"Adjacency matrix saved to {adj_matrix_path}")
    print("Graph definition complete")
    return adj_matrix.values

def normalise_data(train_df, val_df, test_df, output_dir):
    """
    Normalises the data using MinMaxScaler and saves the scaler.

    Args:
        train_df (pd.DataFrame): The training data DataFrame.
        val_df (pd.DataFrame): The validation data DataFrame.
        test_df (pd.DataFrame): The test data DataFrame.
        output_dir (str): The directory path to save the scaler object.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler] or tuple[None, None, None, None]:
            Scaled training, validation, and test data as NumPy arrays, and the fitted scaler object,
            or Nones if normalisation fails.
    """
    # --- PROCESS ---
    # Initialise a MinMaxScaler (feature_range=(0, 1)).
    # Fit the scaler *only* on the train_df numerical data.
    # Transform train_df, val_df, and test_df using the fitted scaler -> train_scaled, val_scaled, test_scaled (NumPy arrays).
    # Save the fitted scaler object (using pickle) for later inverse transformation.
    print("\n--- Step 3: Normalising Data ---")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit ONLY on training data, transform all
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)
    
    # Save the scaler so we can reverse this later for predictions
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Scaler saved to {scaler_path}") 
    print("Data normalisation complete")
    # Return 3 matrices and 1 scaler
    return train_scaled, val_scaled, test_scaled, scaler

def create_sliding_windows(data_scaled, window_size, forecast_horizon):
    """
    Creates sliding windows from scaled data for model input (X) and target (y).

    Args:
        data_scaled (np.ndarray): The scaled time-series data (e.g., train_scaled).
        window_size (int): The number of past time steps to use as input.
        forecast_horizon (int): The number of future time steps to predict.

    Returns:
    tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (Input): Shape (num_samples, window_size, num_nodes, 1)
            - y (Target): Shape (num_samples, forecast_horizon, num_nodes, 1)
            
            Where 'num_samples' is calculated as:
            total_time_steps - window_size - forecast_horizon + 1
    """
    print("\n--- Step 4: Windowing Data ---")
    # --- PROCESS ---
    # Implement function to create sliding windows.
    # This function takes scaled NumPy array data and returns X, y NumPy arrays.
    # Format X with shape (samples, window_size, num_nodes, num_features=1).
    # Format y with shape (samples, forecast_horizon, num_nodes, num_features=1).
    # num_nodes will be data_scaled.shape[1].
    print("\n--- Step 4: Windowing Data ---")
    
    num_samples = data_scaled.shape[0] - window_size - forecast_horizon + 1
    num_nodes = data_scaled.shape[1]
    
    # Initialise arrays
    # Shape: (samples, window_size, nodes, features=1)
    X = np.zeros((num_samples, window_size, num_nodes, 1))
    y = np.zeros((num_samples, forecast_horizon, num_nodes, 1))
    
    for i in range(num_samples):
        # Input: from i to i+window
        X[i, :, :, 0] = data_scaled[i : i + window_size]
        # Target: from i+window to i+window+horizon
        y[i, :, :, 0] = data_scaled[i + window_size : i + window_size + forecast_horizon]
        
    print(f"Generated shapes: X={X.shape}, y={y.shape}")
    return X, y

# ------------------- PDFormer-Specific Functions -------------------

def compute_dtw_matrix(train_df, output_dir):
    """
    Computes and saves the Dynamic Time Warping (DTW) matrix. (PDFormer specific)

    Args:
        train_df (pd.DataFrame): The training data DataFrame.
        output_dir (str): The directory path to save the DTW matrix.

    Returns:
        np.ndarray or None: The calculated DTW matrix or None if calculation fails.
    """
    print("\n--- Step 4a: Computing DTW Matrix (PDFormer) ---")
    # --- PROCESS ---
    # Replicate the logic from PDFormerDataset's _get_dtw method.
    # Calculate the mean of each time series over a daily period (e.g., data_mean).
    # Initialise an empty (num_nodes, num_nodes) matrix for DTW distances.
    # Use a nested loop (and fastdtw) to compute the distance between all pairs of nodes (i, j).
    # Save the resulting dtw_matrix.npy to the output_dir.
    # Return the dtw_matrix.

    # Dull training history to calculate the distance between nodes
    data = train_df.values  # Shape: (time_steps, num_nodes)
    num_nodes = data.shape[1]
    
    # InitialiSe zero matrix
    dtw_matrix = np.zeros((num_nodes, num_nodes))
    
    print(f" - Calculating DTW for {num_nodes} nodes")
    
    # Nested loop to compare every node against every other node
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Calculate distance between Node i and Node j (entire history)
            dist, _ = fastdtw(data[:, i], data[:, j], radius=1)
            dtw_matrix[i, j] = dist
            dtw_matrix[j, i] = dist # Symmetric
            
    print(" - DTW calculation complete.")
    
    # Save
    save_path = os.path.join(output_dir, 'dtw_matrix.npy')
    np.save(save_path, dtw_matrix)
    print(f"Saved dtw_matrix to {save_path}")
    
    return dtw_matrix


def compute_shortest_path_matrices(adj_matrix, output_dir):
    """
    Computes and saves the shortest path hop (sh_mx) and distance (sd_mx) matrices. (PDFormer specific)

    Args:
        adj_matrix (np.ndarray): The base adjacency matrix.
        output_dir (str): The directory path to save the matrices.

    Returns:
        tuple[np.ndarray, np.ndarray] or tuple[None, None]:
            The shortest-path hop matrix (sh_mx) and shortest-path distance matrix (sd_mx).
    """
    print("\n--- Step 4b: Computing Shortest Path Matrices (PDFormer) ---")
    # --- PROCESS ---
    # Replicate the logic from PDFormerDataset's _load_rel method.
    # --- sh_mx (hop matrix) ---
    # Initialise sh_mx from adj_matrix (1s for edges, 511 for no edge, 0 for diagonal).
    # Use the Floyd-Warshall algorithm (3 nested loops) to find the shortest hop count.
    # Save the resulting sh_mx.npy to the output_dir.
    # --- sd_mx (distance matrix) ---
    # This might be computed from a different source in their 'rel' file.
    # For our purpose, we might adapt this or use the adj_matrix itself.
    # Placeholder: save a copy or computed version as sd_mx.npy.
    
    num_nodes = adj_matrix.shape[0]
    
    # Initialize Hop Matrix (sh_mx)
    # 0 for diagonal, 1 for edges, 511 (arbitrary infinity) for non-edges
    sh_mx = np.full((num_nodes, num_nodes), 511, dtype=int)
    
    # Set existing edges to 1
    sh_mx[adj_matrix == 1] = 1
    
    # Set diagonal to 0
    np.fill_diagonal(sh_mx, 0)
    
    # Floyd-Warshall Algorithm to find shortest paths
    # (O(N^3) complexity - fast enough for < 200 nodes)
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                sh_mx[i, j] = min(sh_mx[i, j], sh_mx[i, k] + sh_mx[k, j])
                
    # Save Hop Matrix
    sh_path = os.path.join(output_dir, 'sh_mx.npy')
    np.save(sh_path, sh_mx)
    print(f"Saved sh_mx (Hops) to {sh_path}")

    # Initialise Distance Matrix (sd_mx)
    # In PDFormer, if there are no physical distances, we use the Hop count 
    # or the inverted adjacency weights. Here we will mirror sh_mx for simplicity
    # unless you have a specific physical distance metric.
    sd_mx = sh_mx.astype(float)
    sd_path = os.path.join(output_dir, 'sd_mx.npy')
    np.save(sd_path, sd_mx)
    print(f"Saved sd_mx (Distances) to {sd_path}")

    return sh_mx, sd_mx


def compute_cluster_keys(X_train, n_clusters, output_dir):
    """
    Performs time-series clustering to find pattern keys. (PDFormer specific)

    Args:
        X_train (np.ndarray): The windowed training data (samples, window_size, num_nodes, 1).
        n_clusters (int): The number of clusters (a hyperparameter, e.g., 16).
        output_dir (str): The directory path to save the cluster keys.

    Returns:
        np.ndarray or None: The computed cluster centers (pattern keys).
    """
    print("\n--- Step 4c: Computing Cluster Keys (PDFormer) ---")
    # --- PROCESS ---
    # Replicate the logic from PDFormerDataset's get_data method.
    # Reshape the X_train data into a 2D array suitable for tslearn.
    # Initialise the clustering algorithm (e.g., TimeSeriesKMeans or KShape).
    # Fit the clusterer to the reshaped training data.
    # Get the cluster centers (km.cluster_centers_).
    # Save the pattern_keys.npy to the output_dir.
    # Return the pattern_keys.

    
    # X_train shape: (samples, window_size, num_nodes, 1)
    # ClusterING the "Shapes" of traffic
    # Combine all samples and all nodes to find atomic patterns
    # Reshape to: (Total_Series, Window_Size, 1)
    
    samples, window, nodes, feats = X_train.shape
    X_reshaped = X_train.transpose(0, 2, 1, 3).reshape(-1, window, feats)
    
    # If dataset is huge, we sub-sample for speed (optional)
    # e.g., take random 10,000 samples if X_reshaped.shape[0] > 10000
    if X_reshaped.shape[0] > 10000:
        indices = np.random.choice(X_reshaped.shape[0], 10000, replace=False)
        X_for_clustering = X_reshaped[indices]
    else:
        X_for_clustering = X_reshaped

    print(f" - Clustering {X_for_clustering.shape[0]} time series into {n_clusters} patterns...")
    
    # Use k-Means with Euclidean distance (faster) or DTW (better but slower)
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", max_iter=10, verbose=True)
    km.fit(X_for_clustering)
    
    # The keys are the centroids of the clusters
    pattern_keys = km.cluster_centers_  # Shape: (n_clusters, window, 1)
    
    # Save
    keys_path = os.path.join(output_dir, 'pattern_keys.npy')
    np.save(keys_path, pattern_keys)
    print(f"Saved pattern_keys to {keys_path}")
    
    return pattern_keys


# ------------------- Save Function -------------------

def save_processed_data(output_dir, X_train, y_train, X_val, y_val, X_test, y_test, node_names,
                        adj_matrix=None, dtw_matrix=None, sh_mx=None, sd_mx=None, pattern_keys=None):
    """
    Saves all processed, windowed data and graph Outputs into files.

    Args:
        output_dir (str): The directory path to save the processed files.
        X_train (np.ndarray): Training input sequences.
        y_train (np.ndarray): Training target sequences.
        X_val (np.ndarray): Validation input sequences.
        y_val (np.ndarray): Validation target sequences.
        X_test (np.ndarray): Test input sequences.
        y_test (np.ndarray): Test target sequences.
        node_names (np.ndarray): Array of original column/node names.
        adj_matrix (np.ndarray, optional): Base adjacency matrix.
        dtw_matrix (np.ndarray, optional): Dynamic Time Warping matrix.
        sh_mx (np.ndarray, optional): Shortest-path hop matrix.
        sd_mx (np.ndarray, optional): Shortest-path distance matrix.
        pattern_keys (np.ndarray, optional): Cluster pattern keys.

    Returns:
        None
    """
    print("\n--- Step 5: Saving Processed Data ---")
    # --- PROCESS ---
    # Save the windowed data (X_train, y_train, X_val, y_val, X_test, y_test)
    # as .npz files for convenience (e.g., train.npz, val.npz, test.npz).
    # Save node names (column headers).
    # If PDFormer-specific outputs are provided (not None), save them as .npy files.

    # --- Generic Outputs ---
    
    # 1. Save Windowed Data (.npz)
    # We use 'x' and 'y' keys so the Dataset class can find them easily later
    train_path = os.path.join(output_dir, 'train.npz')
    val_path = os.path.join(output_dir, 'val.npz')
    test_path = os.path.join(output_dir, 'test.npz')
    
    np.savez_compressed(train_path, x=X_train, y=y_train)
    np.savez_compressed(val_path, x=X_val, y=y_val)   # Fixed: x_val -> X_val
    np.savez_compressed(test_path, x=X_test, y=y_test) # Fixed: x_test -> X_test
    
    print(f"Saved .npz files to {output_dir}")

    # 2. Save Node Names
    node_names_path = os.path.join(output_dir, 'node_names.npy')
    np.save(node_names_path, node_names)
    
    # 3. Save Adjacency Matrix
    if adj_matrix is not None:
        np.save(os.path.join(output_dir, 'adj_mx.npy'), adj_matrix)
        print("Saved adj_mx.npy")

    # --- PDFormer-Specific Outputs ---
    
    if dtw_matrix is not None:
        np.save(os.path.join(output_dir, 'dtw_matrix.npy'), dtw_matrix)
        print("Saved dtw_matrix.npy")
        
    if sh_mx is not None:
        np.save(os.path.join(output_dir, 'sh_mx.npy'), sh_mx)
        print("Saved sh_mx.npy")
        
    if sd_mx is not None:
        np.save(os.path.join(output_dir, 'sd_mx.npy'), sd_mx)
        print("Saved sd_mx.npy")
        
    if pattern_keys is not None:
        np.save(os.path.join(output_dir, 'pattern_keys.npy'), pattern_keys)
        print("Saved pattern_keys.npy")
        
    print("Saving complete")

# ------------------- Main function -------------------

def main():
    """
    Main function for the graph preprocessing pipeline.
    """
    # --- Configuration - Argument Parsing ---
    parser = argparse.ArgumentParser(description="Data preprocessing script for graph-based time-series models.")
    parser.add_argument('--pdformer', action='store_true',
                        help="Generate expensive outputs required by PDFormer (DTW, Clustering).")
    args = parser.parse_args()

    # --- Load Configuration from config---
    config = PDFormerConfig()
    
    # 1. Paths (From Config)
    RAW_DATA_FILE = config.raw_data_path
    OUTPUT_DIR = config.processed_data_dir

    # 2. Experimental Settings (From Config)
    TRAIN_SPLIT = config.train_split
    VAL_SPLIT = config.val_split
    
    # Note: Config uses 'input_window' and 'output_window' naming
    WINDOW_SIZE = config.input_window       
    FORECAST_HORIZON = config.output_window 

    # 3. Local Constants (Not currently in Config)
    CORRELATION_THRESHOLD = 0.7
    N_CLUSTERS = 16 

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print(f"--- Configuration ---")
    print(f"Source Data: {RAW_DATA_FILE}")
    print(f"Output Dir:  {OUTPUT_DIR}")
    print(f"Window Size: {WINDOW_SIZE} months")
    print(f"Forecast Horizon: {FORECAST_HORIZON} months")
    print(f"Splits: Train={TRAIN_SPLIT:.0%}, Val={VAL_SPLIT:.0%}, Test={1 - (TRAIN_SPLIT+VAL_SPLIT):.0%}")

    # --- Step 1.1: Load Data ---
    df_raw = load_cyber_threat_data(RAW_DATA_FILE)
    if df_raw is None: 
        print("Aborting: Could not load data.")
        return
    
    # --- Step 1.2: Double Exponential Smoothing ---
    df_smooth = apply_double_exponential_smoothing(df_raw, alpha=0.1, beta=0.1)     

    # --- Step 1.3: Split Data ---
    train_df, val_df, test_df = split_data(df_raw, TRAIN_SPLIT, VAL_SPLIT)

    # --- Step 2: Define Graph Structure ---
    adj_matrix = define_graph_structure(train_df, CORRELATION_THRESHOLD, OUTPUT_DIR)

    # --- Step 3: Normalise Data ---
    train_scaled, val_scaled, test_scaled, scaler = normalise_data(train_df, val_df, test_df, OUTPUT_DIR)

    # --- Step 4: Create Sliding Windows ---
    X_train, y_train = create_sliding_windows(train_scaled, WINDOW_SIZE, FORECAST_HORIZON)
    X_val, y_val = create_sliding_windows(val_scaled, WINDOW_SIZE, FORECAST_HORIZON)
    X_test, y_test = create_sliding_windows(test_scaled, WINDOW_SIZE, FORECAST_HORIZON)
    
    if X_train is None: 
        print("Aborting: Windowing failed.")
        return

    # --- Step 4.1: (Optional) PDFormer-Specific Outputs ---
    dtw_matrix = None
    sh_mx = None
    sd_mx = None
    pattern_keys = None

    if args.pdformer:
        print("\n--- Generating PDFormer-Specific Outputs ---")
        dtw_matrix = compute_dtw_matrix(train_df, OUTPUT_DIR)
        sh_mx, sd_mx = compute_shortest_path_matrices(adj_matrix, OUTPUT_DIR)
        pattern_keys = compute_cluster_keys(X_train, N_CLUSTERS, OUTPUT_DIR)
    else:
        print("\n--- Skipping PDFormer-Specific Outputs (run with --pdformer to generate) ---")

    # --- Step 5: Save Processed Data ---
    node_names = df_raw.columns.values
    save_processed_data(
        OUTPUT_DIR,
        X_train, y_train, X_val, y_val, X_test, y_test,
        node_names,
        adj_matrix=adj_matrix,
        dtw_matrix=dtw_matrix,
        sh_mx=sh_mx,
        sd_mx=sd_mx,
        pattern_keys=pattern_keys
    )

    print("\nPreprocessing script complete.")

if __name__ == '__main__':
    main()