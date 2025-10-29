# Import Standard python library imports
import os
import pickle

# Import Third-party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
    # Create train_df, val_df, test_df pandas DataFrames.
    # Print the date ranges and sizes of each split for verification.
    print("\n--- Step 1: Splitting Data ---")
    # --- CODE ---
    # n = len(df_raw)
    # train_end = int(n * train_split)
    # val_end = int(n * (train_split + val_splitT))
    # train_df = df_raw.iloc[:train_end]
    # val_df = df_raw.iloc[train_end:val_end]
    # test_df = df_raw.iloc[val_end:]
    # print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    print("Data splitting complete.")
    return None, None, None # 3 dfs

def define_graph_structure(train_df, correlation_threshold, output_dir):
    """
    Defines and saves the graph structure (adjacency matrix) based on feature correlations.

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
    print("Graph definition complete.")
    return None # adj_matrix

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
    print("\n--- Step 3: Normalising Data  ---")
    # --- CODE ---
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # train_scaled = scaler.fit_transform(train_df)
    # val_scaled = scaler.transform(val_df)
    # test_scaled = scaler.transform(test_df)
    # scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    # with open(scaler_path, 'wb') as f:
    #     pickle.dump(scaler, f)
    # print(f"Scaler saved to {scaler_path}")
    print("Data normalisation complete.")
    return None, None, None, None # 3 matrices and 1 scaler

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
    # Implement function "create_sliding_windows(data, window_size, horizon, target_indices=None)""
    # This function takes scaled NumPy array data and returns X, y NumPy arrays.
    # Format X with shape (samples, window_size, num_nodes, num_features=1) for models like PDFormer.
    # Format y with shape (samples, forecast_horizon, num_nodes, num_features=1).
    # num_nodes will be the number of columns (e.g., 113).
    # If needed for GAT+Transformer, the function might return X as (samples, window_size, num_nodes * num_features) or similar.
    print("\n--- Step 4: Windowing Data ---")
    # --- CODE ---
    # USING THE SCALED TRAINING DATA HERE
    # X_train, y_train = create_sliding_windows(train_scaled, window_size, forecast_horizon)
    # X_val, y_val = create_sliding_windows(val_scaled, window_size, forecast_horizon)
    # X_test, y_test = create_sliding_windows(test_scaled, window_size, forecast_horizon)
    # print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print("Data windowing complete.")
    return None, None # X,Y

def save_processed_data(output_dir, x_train, y_train, x_val, y_val, x_test, y_test, node_names):
    """
    Saves the processed, windowed data into .npz files and node names.

    Args:
        output_dir (str): The directory path to save the processed files.
        X_train (np.ndarray): Training input sequences.
        y_train (np.ndarray): Training target sequences.
        X_val (np.ndarray): Validation input sequences.
        y_val (np.ndarray): Validation target sequences.
        X_test (np.ndarray): Test input sequences.
        y_test (np.ndarray): Test target sequences.
        node_names (np.ndarray): Array of original column/node names.

    Returns:
        None
    """
    # --- PROCESS ---    
    # Save the windowed data (X_train, y_train, X_val, y_val, X_test, y_test).
    # Saved as .npz files for convenience (e.g., train.npz, val.npz, test.npz).
    # Also save node names (column headers) if needed for interpretation later.
    print("\n--- Step 5: Saving Processed Data ---")
    # --- CODE ---
    # train_path = os.path.join(output_dir, 'train.npz')
    # val_path = os.path.join(output_dir, 'val.npz')
    # test_path = os.path.join(output_dir, 'test.npz')
    # np.savez_compressed(train_path, x=x_train, y=y_train)
    # np.savez_compressed(val_path, x=x_val, y=y_val)
    # np.savez_compressed(test_path, x=x_test, y=y_test)
    # node_names_path = os.path.join(output_dir, 'node_names.npy')
    # np.save(node_names_path, df_raw.columns.values) # Assuming df_raw holds original column names
    # print(f"Processed data saved in {output_dir}")

# ------------------- Main Function-------------------

def main():
    """
    Main function for the graph preprocessing pipeline.
    Serves as a basic test vignette when the script is run directly.
    """
    # --- Configuration - CONSTANTS ---
    SCRIPT_DIR = os.path.dirname(__file__)
    RAW_DATA_FILE = os.path.join(SCRIPT_DIR, '../../Data_Preperation/Cyber_Trend_Forecasting_All.csv')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, '../../Data/Processed_Data_Graph') # graph output

    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.1
    WINDOW_SIZE = 30
    FORECAST_HORIZON = 1
    CORRELATION_THRESHOLD = 0.7

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # --- Step 1: Load Data ---
    df_raw = load_cyber_threat_data(RAW_DATA_FILE)
    if df_raw is None: return

    # --- Step 1.5: Split Data ---
    train_df, val_df, test_df = split_data(df_raw, TRAIN_SPLIT, VAL_SPLIT)
    # TODO: Add check: if train_df is None: return # Or handle error appropriately

    # --- Step 2: Define Graph Structure ---
    # Pass train_df only to avoid data leakage
    adj_matrix = define_graph_structure(train_df, CORRELATION_THRESHOLD, OUTPUT_DIR)
    # TODO: Add check: if adj_matrix is None: return

    # --- Step 3: Normalise Data ---
    train_scaled, val_scaled, test_scaled, scaler = normalise_data(train_df, val_df, test_df, OUTPUT_DIR)
    # TODO: Add check: if train_scaled is None: return

    # --- Step 4: Create Sliding Windows ---
    X_train, y_train = create_sliding_windows(train_scaled, WINDOW_SIZE, FORECAST_HORIZON)
    X_val, y_val = create_sliding_windows(val_scaled, WINDOW_SIZE, FORECAST_HORIZON)
    X_test, y_test = create_sliding_windows(test_scaled, WINDOW_SIZE, FORECAST_HORIZON)
    # TODO: Add check: if X_train is None: return

    # --- Step 5: Save Processed Data ---
    # Ensure df_raw.columns is available or passed appropriately if needed
    node_names = df_raw.columns.values
    save_processed_data(OUTPUT_DIR, X_train, y_train, X_val, y_val, X_test, y_test, node_names)

    print("\nPreprocessing script skeleton complete.")

if __name__ == '__main__':
    main()