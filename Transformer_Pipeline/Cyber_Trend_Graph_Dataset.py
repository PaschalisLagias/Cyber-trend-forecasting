# Standard library imports
import os
import pickle
import sys
from pathlib import Path

# Third-party libraries
import numpy as np
# import torch
from torch.utils.data import Dataset

# Local application imports
# These are only needed for the "on-the-fly" reference code block below
# from .Preprocessing.Load_Data import load_cyber_threat_data
# from .Preprocessing.Cyber_Trend_to_Graph import (
#     split_data, define_graph_structure, normalise_data,
#     create_sliding_windows
# )

class CyberThreatGraphDataset(Dataset):
    """ 
    PyTorch Dataset class for loading pre-processed cyber threat graph data.
    Expects .npz files (train.npz, val.npz, test.npz) and static graph data
    """
    def __init__(self, data_dir, split, output_dir=None):
        """
        Args:
            data_dir (str): Path to directory containing processed files.
            split (str): 'train', 'val', or 'test'.
            output_dir (str, optional): Legacy argument, ignored.
        """
        super().__init__()
        print(f"Initialising dataset for split: {split} from {data_dir}")

        self.data_dir = data_dir
        self.split = split

        # --- 1. Construct File Path ---
        # Expects files named 'train.npz', 'val.npz', or 'test.npz'
        file_name = f"{split}.npz"
        data_path = os.path.join(data_dir, file_name)

        # --- 2. Load Windowed Data ---
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"Loading windowed data from {data_path}...")
        try:
            # Allow pickle for object arrays if needed, x/y will usually works without
            data = np.load(data_path, allow_pickle=True)
            
            # Extract main data arrays
            self.X = data['x']  # Shape: (num_samples, window_size, num_nodes, features)
            self.y = data['y']  # Shape: (num_samples, prediction_horizon, num_nodes, features)
            
            self.num_samples = self.X.shape[0]
            print(f"Loaded {self.num_samples} samples for {split}.")
            
        except KeyError as e:
            raise KeyError(f"The .npz file at {data_path} is missing the key: {e}. Expecting 'x' and 'y'.")
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

        # --- LEGACY: ON-THE-FLY PREPROCESSING (DEPRECATED / FOR REFERENCE ONLY) ---
        #
        # The following code block outlines the original, inefficient "on-the-fly"
        # preprocessing logic. This is kept for reference but is not recommended
        # for use, as all heavy preprocessing should be done offline by
        # Cyber_Trend_to_Graph.py.
        #
        # if raw_data_path is not None and preprocessing_args is not None:
        #     # --- Mode 2: Pre-process On-the-Fly ---
        #     print("Initialising dataset in on-the-fly preprocessing mode...")
        #     # Extract necessary parameters from preprocessing_args.
        #     # Call load_cyber_threat_data(raw_data_path).
        #     # Call split_data(...). Select the appropriate split ('train', 'val', or 'test')
        #     # Call define_graph_structure(...) using the training split part.
        #     # Call normalise_data(...) using train/val/test splits.
        #     # Call create_sliding_windows(...) on the specific scaled split needed.
        #     # Store the resulting X, y, adj_matrix, scaler as instance attributes.
        #     # Note: THIS MODE WILL BE SIGNIFICANTLY SLOWER
        #     print("On-the-fly preprocessing outlined.")
        #
        # --- END LEGACY BLOCK ---

    def __len__(self):
        """
        Returns the total number of samples (time windows) in the dataset.

        Returns:
            int: Number of samples.
        """
        # Return the number of samples stored
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Retrieves the idx-th sample from the dataset.
        
        IMPORTANT: This returns NumPy arrays, NOT tensors. (adhering to PDFormer)
        The conversion to tensors will be handled by the custom collator
        function in the main training script.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - Input sequence (self.X[idx]) as a NumPy array.
                - Target sequence (self.y[idx]) as a NumPy array.
        """
        # Retrieve the input window (self.X[idx]) and target window (self.y[idx]).
        # Return the two NumPy arrays as a tuple.
        return self.X[idx], self.y[idx]

    def get_static_features(self):
        """
        Loads and returns all the static (non-sample-based) graph data
        that are required by the model's constructor.

        Returns:
            dict[str, any]: A dictionary containing all the loaded static data
        """
        print(f"Loading static features from {self.data_dir}...")
        # --- PROCESS ---
        # Initialise an empty dictionary: static_features = {}
        # Load the adjacency matrix (e.g., 'adj_mx.npy') -> static_features['adj_mx'] = ...
        # Load the scaler object (e.g., 'scaler.pkl') -> static_features['scaler'] = ...
        #
        # --- PDFormer-Specific Data ---
        print(f"Loading static features from {self.data_dir}...")
        static_features = {}
        
        # List of potential files (Standard + PDFormer specific)
        files_to_load = {
            'adj_mx': 'adj_mx.npy',
            'dtw_matrix': 'dtw_matrix.npy',
            'sh_mx': 'sh_mx.npy',           # Shortest Path Hop Matrix
            'sd_mx': 'sd_mx.npy',           # Shortest Path Distance Matrix
            'pattern_keys': 'pattern_keys.npy', # Cluster keys
            'scaler': 'scaler.pkl'          # Saved Scaler object
        }

        for key, filename in files_to_load.items():
            path = os.path.join(self.data_dir, filename)
            if os.path.exists(path):
                try:
                    if filename.endswith('.pkl'):
                        with open(path, 'rb') as f:
                            static_features[key] = pickle.load(f)
                    else:
                        static_features[key] = np.load(path)
                    # print(f" - Loaded {key}")
                except Exception as e:
                    print(f"Warning: Failed to load {filename}: {e}")
            else:
                print(f"Note: {filename} not found in directory.")

        return static_features
    
if __name__ == '__main__':
    """
    Example usage demonstrating how to instantiate the Dataset class.
    This serves as a basic test vignette.
    """
    print("\n--- Example Instantiation ---")

    # --- Example: Loading pre-processed data ---
    # This is the standard and recommended way to use this class.
    # Assume pre-processed files exist in '../../data/processed_graph
    
    # Process
    # 1. Add project root to path so we can import the config
    # (Uses the 'sys' and 'Path' you imported at the top)
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    sys.path.append(str(project_root))

    # 2. Import the Config (after sys.path.appended)
    from Transformer_Pipeline.Cyber_Trend_Graph_Config import PDFormerConfig

    # 3. get data
    config = PDFormerConfig()
    data_directory = config.processed_data_dir

    try:
        print("\nAttempting to load 'train' split from pre-processed files...")
        train_dataset_loaded = CyberThreatGraphDataset(data_dir=data_directory, split='train')
        print(f"Loaded dataset length: {len(train_dataset_loaded)}")
    
        # Get a sample (will be NumPy arrays)
        sample_x_np, sample_y_np = train_dataset_loaded[0]
        print(f"Sample X type: {type(sample_x_np)}, Sample Y type: {type(sample_y_np)}")
    
        # Get all static features needed for model init
        static_features = train_dataset_loaded.get_static_features()
        print(f"Loaded static features: {static_features.keys()}") # Should list 'adj_mx', 'dtw_matrix', etc.
    
    except Exception as e:
        print(f"Could not instantiate in load mode: {e}")

    print("\nDataset script skeleton finished.")