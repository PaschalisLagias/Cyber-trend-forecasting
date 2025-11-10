import os
import sys
from pathlib import Path

# Add Transformer_Pipeline to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, TRANSFORMER_DIR)

import numpy as np
import pandas as pd
from Cyber_Trend_Vision_Config import CyberVisionTSConfig
from Preprocessing.Load_Data import load_cyber_threat_data


def split_data(
    df_raw: pd.DataFrame, train_split: float, val_split: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data to train, val, test based on configured train and val ratios.

    Args:
        df_raw: Raw data with date as index.
        train_split: Training split ratio.
        val_split: Validation split ratio.

    Returns:
        tuple of splits (train_df, val_df, test_df).

    """
    # Compute end indices based on configured ratios
    train_end_idx = int(len(df_raw) * train_split)
    val_end_idx = train_end_idx + int(len(df_raw) * val_split)

    # Slice into contiguous, non-overlapping segments
    train_df = df_raw.iloc[:train_end_idx]
    val_df = df_raw.iloc[train_end_idx:val_end_idx]
    test_df = df_raw.iloc[val_end_idx:]
    return train_df, val_df, test_df


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    np.save(output_dir / "train_data.npy", train_df.to_numpy())
    np.save(output_dir / "val_data.npy", val_df.to_numpy())
    np.save(output_dir / "test_data.npy", test_df.to_numpy())


def main(cfg: CyberVisionTSConfig | None = None):
    # TODO: Ask Eshan about verbosity handling for server runs.
    cfg = cfg or CyberVisionTSConfig()
    print("Starting CyberTrendVisionTS preprocessing...")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {cfg.output_dir}")

    df_raw = load_cyber_threat_data(str(cfg.raw_data_path))
    if df_raw is None:
        raise ValueError("Failed to load data.")
    print(f"Loaded data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    print(f"Date range: {df_raw.index.min()} to {df_raw.index.max()}")

    train_df, val_df, test_df = split_data(df_raw, cfg.train_split, cfg.val_split)
    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    save_processed_data(train_df, val_df, test_df, cfg.output_dir)
    print("Saved processed datasets successfully. Done.")


if __name__ == "__main__":
    main()
