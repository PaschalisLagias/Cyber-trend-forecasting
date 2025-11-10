import os
import sys
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, TRANSFORMER_DIR)

from Cyber_Trend_Vision_Config import CyberVisionTSConfig
from Preprocessing.Cyber_Trend_to_Image import split_data
from Preprocessing.Load_Data import load_cyber_threat_data


class CyberTrendImageDataset(Dataset):
    """PyTorch Dataset for multivariate time series forecasting with VisionTS."""

    def __init__(
        self,
        data: np.ndarray | pd.DataFrame,
        context_window_size: int,
        pred_window_size: int,
        is_train: bool = True,
        transform: Callable | None = None,
    ) -> None:
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        self.data = torch.from_numpy(data).float()
        self.context_window_size = context_window_size
        self.pred_window_size = pred_window_size
        self.is_train = is_train
        self.transform = transform

        self.num_features = self.data.shape[1]
        self.num_samples = len(self.data) - (self.context_window_size + self.pred_window_size) + 1
        if self.num_samples <= 0:
            raise ValueError(
                f"{self.num_samples=} too small given {self.context_window_size=} and {self.pred_window_size=}.",
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ctx_end = idx + self.context_window_size
        pred_end = ctx_end + self.pred_window_size
        x = self.data[idx:ctx_end]
        y = self.data[ctx_end:pred_end]
        sample = {"input": x, "target": y}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __repr__(self) -> str:
        data_shape = tuple(self.data.shape)
        context_window_shape = (self.context_window_size, self.num_features)
        target_window_shape = (self.pred_window_size, self.num_features)
        return f"{self.__class__.__name__}({data_shape=}, {context_window_shape=}, {target_window_shape=})"


if __name__ == "__main__":
    print("\n--- Example Instantiation ---")
    cfg = CyberVisionTSConfig()
    data_raw = load_cyber_threat_data()
    if data_raw is None:
        raise ValueError("Failed to load data.")
    train_data, val_data, test_data = split_data(data_raw, cfg.train_split, cfg.val_split)

    dataset = CyberTrendImageDataset(
        data=train_data,
        context_window_size=cfg.context_window_size,
        pred_window_size=cfg.pred_window_size,
    )
    print(dataset)
    print(f"Dataset length: {len(dataset)} samples")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Input shape: {sample['input'].shape}, Target shape: {sample['target'].shape}")
    print(f"Input type: {type(sample['input'])}, Target type: {type(sample['target'])}")
