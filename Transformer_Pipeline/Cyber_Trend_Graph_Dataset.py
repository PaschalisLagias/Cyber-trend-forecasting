# Standard library imports
import os
import pickle

# Third-party libraries
import numpy as np
import torch
from torch.utils.data import Dataset

# Local application imports
# Imports only necessary if preprocessing done on-the-fly
from .Preprocessing.Load_Data import load_cyber_threat_data
from .Preprocessing.Cyber_Trend_to_Graph import (
    split_data, define_graph_structure, normalise_data,
    create_sliding_windows
)

class CyberThreatGraphDataset(Dataset):
    """
    PyTorch Dataset class for loading cyber threat graph data.

    Can operate in two modes:
    1. Load pre-processed data: Reads .npz and .npy files from a local directory.
    2. Pre-process on-the-fly: Loads raw data and runs the preprocessing pipeline.
    """
    def __init__(self, data_dir=None, split=None, raw_data_path=None, preprocessing_args=None):
        """
        Initialises the dataset.

        Args:
            data_dir (str, optional): Path to the directory containing pre-processed
                                      files (e.g., train.npz, val.npz, adj_mx.npy, scaler.pkl).
                                      Required if loading pre-processed data. Defaults to None.
            split (str, optional): Which data split to load ('train', 'val', or 'test').
                                   Required if loading pre-processed data. Defaults to None.
            raw_data_path (str, optional): Path to the raw CSV data file.
                                           Required if preprocessing on-the-fly. Defaults to None.
            preprocessing_args (dict, optional): Dictionary containing arguments needed for
                                                 on-the-fly preprocessing (e.g., {'window_size': 30,
                                                 'horizon': 1, 'threshold': 0.7, 'train_split': 0.7, ...}).
                                                 Required if preprocessing on-the-fly. Defaults to None.

        Raises:
            ValueError: If required arguments for a chosen mode are missing or incorrect.
        """
        super().__init__()

        # --- Input Validation ---
        # Check if arguments for one and only one mode are provided.
        # Raise ValueError if arguments are missing or incorrect.
        # Example check:
        # load_mode = data_dir is not None and split is not None
        # preprocess_mode = raw_data_path is not None and preprocessing_args is not None
        # if load_mode == preprocess_mode: # Either both true or both false
        #     raise ValueError("Provide arguments for either loading pre-processed data OR on-the-fly preprocessing, not both or neither.")

        # --- Data Loading / Processing ---
        if data_dir is not None and split is not None:
            # --- Mode 1: Load Pre-processed Data ---
            print(f"Initialising dataset in load mode for split: {split} from {data_dir}")
            # Construct file paths for the specific split (e.g., train.npz) and common files (adj_mx.npy).
            # Load the .npz file for the specified split (contains 'x' and 'y').
            # Load the adjacency matrix (.npy file).
            # Load the scaler object (.pkl file).
            # Store loaded data (X, y, adj_matrix, scaler) as instance attributes (e.g., self.X, self.y, etc.).
            # Ensure data types are appropriate (e.g., adj_matrix might be needed as a tensor later).
            print("Data loaded from file.")

        elif raw_data_path is not None and preprocessing_args is not None:
            # --- Mode 2: Pre-process On-the-Fly ---
            print("Initialising dataset in on-the-fly preprocessing mode...")
            # Extract necessary parameters from preprocessing_args.
            # Call load_cyber_threat_data(raw_data_path).
            # Call split_data(...). Select the appropriate split ('train', 'val', or 'test') based on an argument in preprocessing_args.
            # Call define_graph_structure(...) using the training split part.
            # Call normalise_data(...) using train/val/test splits.
            # Call create_sliding_windows(...) on the specific scaled split needed.
            # Store the resulting X, y, adj_matrix, scaler as instance attributes.
            # Note: THIS MODE WILL BE SIGNIFICANTLY SLOWER
            print("On-the-fly preprocessing outlined.")

        else:
             # This case should be caught by input validation, but added for completeness.
             raise ValueError("Invalid combination of arguments provided for dataset initialisation.")

        # --- Finalise Setup ---
        # Store essential information like number of samples, window size, etc.
        # self.num_samples = ... # Calculate based on loaded self.X
        # self.window_size = ...
        # self.horizon = ...
        # self.num_nodes = ...
        # self.num_features = ...

    def __len__(self):
        """
        Returns the total number of samples (time windows) in the dataset.

        Returns:
            int: Number of samples.
        """
        # Return the number of samples stored (e.g., self.X.shape[0] or calculated in main).
        return 0 # length

    def __getitem__(self, idx):
        """
        Retrieves the idx-th sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Input sequence tensor (shape: [window_size, num_nodes, num_features])
                - Target sequence tensor (shape: [forecast_horizon, num_nodes, num_features])
        """
        # Retrieve the input window (self.X[idx]) and target window (self.y[idx]).
        # Convert the NumPy arrays to PyTorch tensors (torch.Tensor).
        # Ensure correct data types (e.g., torch.float32).
        # Return the input tensor and target tensor as a tuple.
        # Note: Adjacency matrix is usually handled globally or passed differently, not typically returned per sample.
        return None, None # Input, Target


if __name__ == '__main__':
    """
    Instantiate the Dataset class in either mode.
    This serves as a basic test vignette.
    """
    print("\n--- Creating Graph Dataset ---")

    # --- Mode 1: Loading pre-processed graph data ---
    # Assumes pre-processed graph data exist in '../../Data/Processed_Data_Graph'
    # data_directory = '../../Data/Processed_Data_Graph'
    # try:
    #     print("\nAttempting to load 'train' split from pre-processed files...")
    #     train_dataset_loaded = CyberThreatGraphDataset(data_dir=data_directory, split='train')
    #     print(f"Loaded dataset length: {len(train_dataset_loaded)}")
    #     # Check a sample
    #     sample_x, sample_y = train_dataset_loaded[0]
    #     print(f"Sample X shape: {sample_x.shape}, Sample Y shape: {sample_y.shape}") # Should print tensor shapes
    # except Exception as e:
    #     print(f"Could not instantiate in load mode: {e}")

    # --- Mode 2: Pre-processing data on-the-fly ---
    # Assumes raw data file exists at '../../Data_Preperation/Cyber_Trend_Forecasting_All.csv'
    # raw_file_path = '../../Data_Preperation/Cyber_Trend_Forecasting_All.csv'
    # preproc_args = {
    #     'window_size': 30, 'horizon': 1, 'threshold': 0.7,
    #     'train_split': 0.7, 'val_split': 0.1,
    #     'output_dir_temp': './temp_on_the_fly_output', # Need temporary dir for scaler/adj
    #     'target_split': 'val' # Example: process only to get the validation set
    # }
    # try:
    #     print("\nAttempting to create 'val' split by pre-processing on-the-fly...")
    #     # Ensure necessary preprocessing functions are imported at the top if testing this
    #     val_dataset_processed = CyberThreatGraphDataset(raw_data_path=raw_file_path, preprocessing_args=preproc_args)
    #     print(f"Processed dataset length: {len(val_dataset_processed)}")
    #     # Get a sample
    #     sample_x_p, sample_y_p = val_dataset_processed[0]
    #     print(f"Sample X shape: {sample_x_p.shape}, Sample Y shape: {sample_y_p.shape}") # Should print tensor shapes
    # except Exception as e:
    #     print(f"Could not instantiate in on-the-fly mode: {e}")

    print("\nDataset Ready.")
