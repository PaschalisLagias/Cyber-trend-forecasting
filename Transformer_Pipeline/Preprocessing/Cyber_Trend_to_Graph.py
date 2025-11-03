# Import Standard python library imports
import os
import pickle
import argparse # Added for command-line flag

# Import Third-party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Imports for PDFormer transformations (uncomment when implementing)
# from fastdtw import fastdtw
# from tslearn.clustering import TimeSeriesKMeans

# Local imports - using project filename conventions
from .Load_Data import load_cyber_threat_data

# ------------------- Preprocessing Functions -------------------

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
    print("\n--- Step 1: Splitting Data ---")
    # --- CODE ---
    # n = len(df_raw)
    # train_end = int(n * train_split)
    # val_end = int(n * (train_split + val_split))
    # train_df = df_raw.iloc[:train_end]
    # val_df = df_raw.iloc[train_end:val_end]
    # test_df = df_raw.iloc[val_end:]
    # print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    print("Data splitting outlined.")
    # We return 3 dataframes 
    return df_raw, df_raw, df_raw # Returning df_raw 3x as placeholder to avoid errors in main()

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
    # --- CODE ---
    # corr_matrix = train_df.corr()
    # adj_matrix = (np.abs(corr_matrix) > correlation_threshold).astype(int)
    # np.fill_diagonal(adj_matrix.values, 1)
    # adj_matrix_path = os.path.join(output_dir, 'adj_mx.npy')
    # np.save(adj_matrix_path, adj_matrix.values)
    # print(f"Adjacency matrix saved to {adj_matrix_path}")
    print("Graph definition outlined.")
    # We return a matrix 
    return np.zeros((train_df.shape[1], train_df.shape[1])) # Placeholder matrix

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
    # --- CODE ---
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # train_scaled = scaler.fit_transform(train_df)
    # val_scaled = scaler.transform(val_df)
    # test_scaled = scaler.transform(test_df)
    # scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    # with open(scaler_path, 'wb') as f:
    #     pickle.dump(scaler, f)
    # print(f"Scaler saved to {scaler_path}")
    print("Data normalisation outlined.")
    # Return 3 matrices and 1 scaler
    return train_df.values, val_df.values, test_df.values, None # Placeholders

def create_sliding_windows(data_scaled, window_size, forecast_horizon):
    """
    Creates sliding windows from scaled data for model input (X) and target (y).

    Args:
        data_scaled (np.ndarray): The scaled time-series data (e.g., train_scaled).
        window_size (int): The number of past time steps to use as input.
        forecast_horizon (int): The number of future time steps to predict.

    Returns:
        tuple[np.ndarray, np.ndarray] or tuple[None, None]:
            Input sequences (X) and target sequences (y) as NumPy arrays, or Nones if windowing fails.
            Expected shape for X: (samples, window_size, num_nodes, num_features=1)
            Expected shape for y: (samples, forecast_horizon, num_nodes, num_features=1)
    """
    print("\n--- Step 4: Windowing Data ---")
    # --- PROCESS ---
    # Implement function to create sliding windows.
    # This function takes scaled NumPy array data and returns X, y NumPy arrays.
    # Format X with shape (samples, window_size, num_nodes, num_features=1).
    # Format y with shape (samples, forecast_horizon, num_nodes, num_features=1).
    # num_nodes will be data_scaled.shape[1].
    print("Data windowing outlined.")
    # --- CODE ---
    # Example placeholder return (replace with actual arrays)
    # num_samples = data_scaled.shape[0] - window_size - forecast_horizon + 1
    # num_nodes = data_scaled.shape[1]
    # X_shape = (num_samples, window_size, num_nodes, 1)
    # y_shape = (num_samples, forecast_horizon, num_nodes, 1)
    # print(f"Calculated shapes: X {X_shape}, y {y_shape}")
    return np.zeros((100, window_size, data_scaled.shape[1], 1)), np.zeros((100, forecast_horizon, data_scaled.shape[1], 1)) # Placeholder arrays

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
    print("DTW matrix computation outlined.")
    return None # matrix

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
    print("Shortest path matrix computation outlined.")
    return None, None # 2 matrices

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
    print("Cluster key computation outlined.")
    return None # keys

# ------------------- Save Function -------------------

def save_processed_data(output_dir, X_train, y_train, X_val, y_val, X_test, y_test, node_names,
                        adj_matrix=None, dtw_matrix=None, sh_mx=None, sd_mx=None, pattern_keys=None):
    """
    Saves all processed, windowed data and graph artifacts into files.

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
    # If PDFormer-specific artifacts are provided (not None), save them as .npy files.

    # --- CODE ---
    # --- Generic Artifacts ---
    # train_path = os.path.join(output_dir, 'train.npz')
    # val_path = os.path.join(output_dir, 'val.npz')
    # test_path = os.path.join(output_dir, 'test.npz')
    # np.savez_compressed(train_path, x=X_train, y=y_train)
    # np.savez_compressed(val_path, x=x_val, y=y_val)
    # np.savez_compressed(test_path, x=x_test, y=y_test)
    # node_names_path = os.path.join(output_dir, 'node_names.npy')
    # np.save(node_names_path, node_names)
    # if adj_matrix is not None:
    #   np.save(os.path.join(output_dir, 'adj_mx.npy'), adj_matrix)
    # print(f"Generic processed data saved in {output_dir}")

    # --- PDFormer-Specific Artifacts ---
    # if dtw_matrix is not None:
    #     np.save(os.path.join(output_dir, 'dtw_matrix.npy'), dtw_matrix)
    #     print("Saved dtw_matrix.npy")
    # if sh_mx is not None:
    #     np.save(os.path.join(output_dir, 'sh_mx.npy'), sh_mx)
    #     print("Saved sh_mx.npy")
    # if sd_mx is not None:
    #     np.save(os.path.join(output_dir, 'sd_mx.npy'), sd_mx)
    #     print("Saved sd_mx.npy")
    # if pattern_keys is not None:
    #     np.save(os.path.join(output_dir, 'pattern_keys.npy'), pattern_keys)
    #     print("Saved pattern_keys.npy")
    print("Saving processed data outlined.")

# ------------------- Main Orchestration -------------------

def main():
    """
    Main function for the graph preprocessing pipeline.
    Serves as a basic test vignette when the script is run directly.
    """
    # --- Configuration - Argument Parsing ---
    parser = argparse.ArgumentParser(description="Data preprocessing script for graph-based time-series models.")
    parser.add_argument('--pdformer', action='store_true',
                        help="Set this flag to generate the additional, expensive artifacts required by the PDFormer model (e.g., DTW matrix, cluster keys).")
    args = parser.parse_args()

    # --- Configuration - CONSTANTS ---
    SCRIPT_DIR = os.path.dirname(__file__)
    # Note: Using the corrected path based on our earlier discussion
    RAW_DATA_FILE = os.path.join(SCRIPT_DIR, '../../Data_Preparation/Cyber_Trend_Forecasting_All.csv')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, '../../data/processed_graph') # Corrected output path

    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.1
    WINDOW_SIZE = 30
    FORECAST_HORIZON = 1
    CORRELATION_THRESHOLD = 0.7
    N_CLUSTERS = 16 # Example hyperparameter for PDFormer clustering

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # --- Step 1: Load Data ---
    df_raw = load_cyber_threat_data(RAW_DATA_FILE)
    if df_raw is None: return

    # --- Step 1.5: Split Data ---
    train_df, val_df, test_df = split_data(df_raw, TRAIN_SPLIT, VAL_SPLIT)
    if train_df is None: return # Check if split failed

    # --- Step 2: Define Graph Structure ---
    adj_matrix = define_graph_structure(train_df, CORRELATION_THRESHOLD, OUTPUT_DIR)
    if adj_matrix is None: return # Check if graph failed

    # --- Step 3: Normalise Data ---
    train_scaled, val_scaled, test_scaled, scaler = normalise_data(train_df, val_df, test_df, OUTPUT_DIR)
    if train_scaled is None: return # Check if normalise failed

    # --- Step 4: Create Sliding Windows ---
    X_train, y_train = create_sliding_windows(train_scaled, WINDOW_SIZE, FORECAST_HORIZON)
    X_val, y_val = create_sliding_windows(val_scaled, WINDOW_SIZE, FORECAST_HORIZON)
    X_test, y_test = create_sliding_windows(test_scaled, WINDOW_SIZE, FORECAST_HORIZON)
    if X_train is None: return # Check if windowing failed

    # --- Step 4.5: (Optional) PDFormer-Specific Artifacts ---
    dtw_matrix = None
    sh_mx = None
    sd_mx = None
    pattern_keys = None

    if args.pdformer:
        print("\n--- Generating PDFormer-Specific Artifacts (this may take a while) ---")
        dtw_matrix = compute_dtw_matrix(train_df, OUTPUT_DIR)
        sh_mx, sd_mx = compute_shortest_path_matrices(adj_matrix, OUTPUT_DIR)
        pattern_keys = compute_cluster_keys(X_train, N_CLUSTERS, OUTPUT_DIR)
        print("--- PDFormer artifact generation outlined ---")
    else:
        print("\n--- Skipping PDFormer-Specific Artifacts (run with --pdformer to generate) ---")

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

    print("\nPreprocessing script skeleton complete.")

if __name__ == '__main__':
    main()