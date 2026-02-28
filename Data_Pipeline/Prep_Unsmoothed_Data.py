"""
Phase 1: Data Preparation Pipeline for B-MTGNN
==============================================

This module handles the ingestion and formatting of the raw, unsmoothed
cyber threat dataset (extending to Jan-25). 

It reads the 'column_mapping.csv' generated in Phase 0 to safely translate
the legacy node names to the new feature names. It then extracts those 
exact 123 columns from the new dataset, enforces the original legacy order,
and exports a pure numeric, tab-separated text file.

It saves the output to the Processed_Data folder to ensure the original 
author's legacy workspace remains completely untouched.

Usage:
    Run this script directly from the Data_Pipeline directory:
    $ python Prep_Unsmoothed_Data.py
"""

import sys
from pathlib import Path
import pandas as pd

# --- Path Configuration ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Input paths
INPUT_CSV_PATH = PROJECT_ROOT / "Data_Preparation" / "Cyber_Trend_Forecasting_All.csv"
LEGACY_HEADER_PATH = PROJECT_ROOT / "B-MTGNN" / "data" / "sm_data_g.csv"
MAPPING_CSV_PATH = CURRENT_DIR / "column_mapping.csv"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "Processed_Data" / "B-MTGNN"
OUTPUT_TXT_PATH = OUTPUT_DIR / "sm_data.txt"


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

    # 6. Slice and Order the dataset
    try:
        df_filtered = df_new[ordered_new_cols]
        print(f"Filtered dataset shape: {df_filtered.shape} (Months x Columns)")
    except KeyError as e:
        print(f"CRITICAL ERROR: Mapped column missing from new dataset. {e}")
        sys.exit(1)

    # 7. Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 8. Export to the specific tab-separated format required by B-MTGNN
    print(f"Exporting formatted data to: {OUTPUT_TXT_PATH.relative_to(PROJECT_ROOT)}")
    try:
        # We use to_csv to write without headers or row indices, separated by tabs.
        # Using float format ensures strict compatibility with PyTorch tensor loading.
        df_filtered.to_csv(OUTPUT_TXT_PATH, sep='\t', header=False, index=False, float_format='%.6f')
        print("\nSUCCESS: Phase 1 Data preparation complete.")
        print(f"File successfully saved to: {OUTPUT_TXT_PATH}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Failed to save data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    prepare_data()