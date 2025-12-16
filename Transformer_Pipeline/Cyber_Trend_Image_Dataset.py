from __future__ import annotations

import os
import sys
from typing import Callable, Optional
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, TRANSFORMER_DIR)


class CyberTrendImageDataset(Dataset):
    """
    PyTorch Dataset for VisionTS-based multivariate time series forecasting.

    VisionTS expects input tensors of shape: (batch, context_len, nvars)
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data_dir: Path to directory containing preprocessed .npz files.
                      If provided, loads from preprocessed files.
            split: Which split to load ('train', 'val', or 'test').
                   Only used when data_dir is provided.
            transform: Optional transform to apply to samples.
        """
        super().__init__()
        self.transform = transform
        self.data_dir = data_dir
        self.split = split

        if data_dir is not None:
            self._init_from_preprocessed(data_dir, split)
        else:
            raise ValueError(
                "Must provide 'data_dir' (for preprocessed files)."
            )

    def _init_from_preprocessed(self, data_dir: str, split: str) -> None:
        print(f"Initializing VisionTS dataset for split: {split} from {data_dir}")

        # Construct file path
        file_name = f"{split}.npz"
        data_path = os.path.join(data_dir, file_name)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load windowed data
        print(f"Loading windowed data from {data_path}...")
        try:
            data = np.load(data_path, allow_pickle=True)
            self.X = data["x"]  # Shape: (num_samples, context_len, num_features)
            self.y = data["y"]  # Shape: (num_samples, pred_len, num_features)

            self.num_samples = self.X.shape[0]
            self.context_window_size = self.X.shape[1]
            self.pred_window_size = self.y.shape[1]
            self.num_features = self.X.shape[2]

            print(f"Loaded {self.num_samples} samples for {split}.")
            print(f"  X shape: {self.X.shape} (samples, context_len, nvars)")
            print(f"  y shape: {self.y.shape} (samples, pred_len, nvars)")

        except KeyError as e:
            raise KeyError(f"The .npz file at {data_path} is missing key: {e}. Expected 'x' and 'y' keys.")

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retrieve the idx-th sample from the dataset.

        Returns:
            dict with 'input' and 'target' tensors:
                - input: Shape (context_len, num_features)
                - target: Shape (pred_len, num_features)
        """
        # Preprocessed mode: load from arrays
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.y[idx]).float()

        sample = {"input": x, "target": y}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_static_features(self) -> dict:
        """
        Load and return static features (scaler, metadata) for model initialization.

        This method provides consistency with the Graph pipeline's Dataset interface.
        For VisionTS, the scaler is saved for reference but NOT applied to data
        (VisionTS handles normalisation internally).

        Returns:
            dict containing:
                - scaler: Reference MinMaxScaler (not applied to data)
                - metadata: Preprocessing parameters
                - feature_names: Column/feature names
        """
        if self.data_dir is None:
            print("Warning: get_static_features() called but no data_dir set.")
            return {}

        print(f"Loading static features from {self.data_dir}...")
        static_features = {}

        # Files to potentially load
        files_to_load = {
            "scaler": "scaler.pkl",
            "metadata": "metadata.pkl",
            "feature_names": "feature_names.npy",
        }

        for key, filename in files_to_load.items():
            path = os.path.join(self.data_dir, filename)
            if os.path.exists(path):
                try:
                    if filename.endswith(".pkl"):
                        with open(path, "rb") as f:
                            static_features[key] = pickle.load(f)
                    else:
                        static_features[key] = np.load(path, allow_pickle=True)
                    print(f"  - Loaded {key}")
                except Exception as e:
                    print(f"Warning: Failed to load {filename}: {e}")
            else:
                print(f"Note: {filename} not found in directory.")

        return static_features

    def __repr__(self) -> str:
        """String representation of the dataset."""
        if self.X is not None:
            data_shape = self.X.shape
        else:
            data_shape = tuple(self.raw_data.shape)

        context_shape = (self.context_window_size, self.num_features)
        target_shape = (self.pred_window_size, self.num_features)

        return f"{self.__class__.__name__}(samples={self.num_samples}, context={context_shape}, target={target_shape})"


# ------------------- Example Usage -------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add parent directory to path for imports
    SCRIPT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPT_DIR))

    print("\n" + "=" * 60)
    print("CyberTrendImageDataset - Example Usage")
    print("=" * 60)

    # Try to load from preprocessed files
    processed_dir = SCRIPT_DIR.parent / "Processed_Data" / "vision"

    if processed_dir.exists() and (processed_dir / "train.npz").exists():
        print("\n--- Mode 1: Loading from Preprocessed Files ---")
        dataset = CyberTrendImageDataset(data_dir=str(processed_dir), split="train")
        print(f"Dataset: {dataset}")

        # Get a sample
        sample = dataset[0]
        print(f"\nSample keys: {list(sample.keys())}")
        print(f"Input shape: {sample['input'].shape}")
        print(f"Target shape: {sample['target'].shape}")

        # Get static features
        static = dataset.get_static_features()
        print(f"\nStatic features: {list(static.keys())}")
    else:
        print("\nWarning: Preprocessed files not found. Skipping preprocessed mode.")
