"""
Phase 1: Data Preparation Pipeline for B-MTGNN and VisionTS++
=============================================================

This module handles the ingestion and formatting of the raw, unsmoothed
cyber threat dataset (extending to Dec-25). 

It reads the 'column_mapping.csv' generated in Phase 0 to safely translate
the legacy node names to the new feature names. It extracts the required 123 
columns, enforces the original legacy order, and applies bounding to correct 
upstream smoothing artifacts.

It exports two formats:
1. sm_data.txt: A headerless, tab-separated numeric matrix for the legacy B-MTGNN.
2. Mark3_Clipped_Data.csv: A standard CSV retaining headers and temporal indices for VisionTS++.

Usage:
    Run this script directly from the Data_Pipeline directory:
    $ python Prep_Unsmoothed_Data.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# --- Path Configuration ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Input paths
# INPUT_CSV_PATH = PROJECT_ROOT / "Data_Preparation" / "Cyber_Trend_Forecasting_All.csv"
INPUT_CSV_PATH = PROJECT_ROOT / "Data_Preparation" / "CSIS" / "csis_output_20260404-01.csv"
LEGACY_HEADER_PATH = PROJECT_ROOT / "B-MTGNN" / "data" / "sm_data_g.csv"
MAPPING_CSV_PATH = CURRENT_DIR / "column_mapping.csv"

# Output paths
OUTPUT_DIR_BMTGNN = PROJECT_ROOT / "Processed_Data" / "B-MTGNN"
OUTPUT_DIR_VISION = PROJECT_ROOT / "Processed_Data" / "VisionTS"

OUTPUT_TXT_PATH = OUTPUT_DIR_BMTGNN / "sm_data.txt"
OUTPUT_CSV_PATH = OUTPUT_DIR_VISION / "Mark3_Clipped_Data.csv"

# --- SUCCEEDING CODE ---
def prepare_data():
    """
    Loads the new CSV dataset, filters it using the column mapping,
    enforces the strict legacy column order, and exports the tensor matrix.
    """
    print("=" * 60)
    print("--- Initiating Phase 1: B-MTGNN Data Preparation ---")
    print("=" * 60)

    # 1. Verify files exist
    for path in [INPUT_CSV_PATH, LEGACY_HEADER_PATH, MAPPING_CSV_PATH]:
        if not path.exists():
            print(f"CRITICAL ERROR: Cannot find required file at {path}")
            sys.exit(1)

    # 2. Load the legacy headers to establish the STRICT order required
    legacy_df = pd.read_csv(LEGACY_HEADER_PATH, nrows=0)
    legacy_cols = [c for c in legacy_df.columns if 'date' not in c.lower() and 'month' not in c.lower()]

    # 3. Load the mapping generated in Phase 0
    mapping_df = pd.read_csv(MAPPING_CSV_PATH)
    mapping_dict = dict(zip(mapping_df['Legacy_Node_Name'], mapping_df['New_Dataset_Column']))
    
    # Apply manual patch for the one missing column identified in Phase 0
    mapping_dict["Solution_MACHINE LEARNING_Mentions"] = "Solution_ML/DL_Papers"

    # 4. Build the ordered list of new column names
    ordered_new_cols = []
    for old_col in legacy_cols:
        new_col = mapping_dict.get(old_col)
        if not new_col:
            print(f"CRITICAL ERROR: No mapping found for legacy node '{old_col}'")
            sys.exit(1)
        ordered_new_cols.append(new_col)

    # 5. Load the new, massive dataset
    print(f"Loading raw dataset from: {INPUT_CSV_PATH.relative_to(PROJECT_ROOT)}")
    try:
        df_new = pd.read_csv(INPUT_CSV_PATH)
        print(f"Original dataset shape: {df_new.shape} (Months x Columns)")
    except Exception as e:
        print(f"Failed to read new CSV: {e}")
        sys.exit(1)

    # 6. Extract temporal index and slice the dataset
    try:
        # Identify the temporal column to retain for VisionTS++
        temporal_cols = [c for c in df_new.columns if 'date' in c.lower() or 'month' in c.lower()]
        temporal_col_name = temporal_cols[0] if temporal_cols else df_new.columns[0]
        temporal_series = df_new[temporal_col_name]

        # Extract the mapped columns using a copy to prevent SettingWithCopy warnings
        df_filtered = df_new[ordered_new_cols].copy()
        print(f"Filtered dataset shape: {df_filtered.shape} (Months x Columns)")
    except KeyError as e:
        print(f"CRITICAL ERROR: Mapped column missing from new dataset. {e}")
        sys.exit(1)

    # Eliminate negative value artifacts caused by upstream double exponential smoothing
    print("Applying np.clip to sanitize data (setting minimum bound to 0.0)...")
    df_filtered = np.clip(df_filtered, a_min=0.0, a_max=None)    

    # Construct the secondary DataFrame for VisionTS++
    df_vision = pd.concat([temporal_series, df_filtered], axis=1)

    # 7. Ensure output directories exist
    OUTPUT_DIR_BMTGNN.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_VISION.mkdir(parents=True, exist_ok=True)

    # 8. Execute dual exports
    print(f"Exporting legacy B-MTGNN matrix to: {OUTPUT_TXT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Exporting VisionTS++ CSV to: {OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT)}")
    try:
        # Export 1: Legacy B-MTGNN (headerless, tab-separated, strictly float matrix)
        df_filtered.to_csv(OUTPUT_TXT_PATH, sep='\t', header=False, index=False, float_format='%.6f')
        
        # Export 2: VisionTS++ (standard CSV, retains headers and temporal column)
        df_vision.to_csv(OUTPUT_CSV_PATH, index=False)
        
        print("\nSUCCESS: Phase 1 Data preparation complete.")
        print(f"Legacy Matrix saved to: {OUTPUT_TXT_PATH}")
        print(f"VisionTS++ CSV saved to: {OUTPUT_CSV_PATH}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Failed to save dual exports: {e}")
        sys.exit(1)

if __name__ == "__main__":
    prepare_data()