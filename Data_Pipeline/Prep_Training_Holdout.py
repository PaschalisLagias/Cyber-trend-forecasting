"""
Data Alignment and Training Window Extraction
=============================================

This script verifies alignment between the Hackmageddon monthly 
incident records and the complete cybertrend dataset (v2.1).
It confirms that all attack column names match exactly across both
files, and then extracts the historical training window spanning from
July 2011 to December 2023 for validation modelling.
"""

import sys
from pathlib import Path
import pandas as pd

# Import paths dynamically from the central Config file
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Config.Paths import (
    HACK_NOI_MONTHLY_V2 as HACKMAGEDDON_PATH,
    V2_1_DATA_CSV as V2_1_PATH,
    TRAIN_HACK_CSV as OUTPUT_HACK_TRAIN,
    TRAIN_V2_1_CSV as OUTPUT_V2_TRAIN
)


def verify_and_slice_data():
    """
    Loads datasets, confirms header alignment, and exports the training
    window required for the 2024 holdout validation.
    """
    print("Loading Hackmageddon incident data and v2.1 social dataset...")
    df_hackmageddon = pd.read_csv(HACKMAGEDDON_PATH)
    df_social = pd.read_csv(V2_1_PATH)

    # Extract column names while excluding the initial date header
    hackmageddon_columns = set(df_hackmageddon.columns[1:])
    social_columns = set(df_social.columns[1:])

    # Identify any discrepancies in column naming between the two files
    missing_from_social = hackmageddon_columns - social_columns

    if missing_from_social:
        print("ERROR: Column mismatch detected.")
        print(f"Columns present in Hackmageddon but missing in v2.1: {missing_from_social}")
        sys.exit(1)
    else:
        print("SUCCESS: All Hackmageddon attack columns align with v2.1.")

    # Standardise date columns to datetime objects to ensure clean chronological slicing.
    # Hackmageddon uses 'Attack-Country' or 'Date' depending on the export, while v2.1 uses 'Date'.
    date_col_hack = df_hackmageddon.columns[0]
    date_col_social = df_social.columns[0]

    # Attempt standard MM/YYYY parsing first, with a fallback to MMM-YY formatting
    df_hackmageddon['parsed_date'] = pd.to_datetime(
        df_hackmageddon[date_col_hack], format='%m/%Y', errors='coerce'
    )
    if df_hackmageddon['parsed_date'].isna().all():
        df_hackmageddon['parsed_date'] = pd.to_datetime(
            df_hackmageddon[date_col_hack], format='%b-%y', errors='coerce'
        )

    df_social['parsed_date'] = pd.to_datetime(
        df_social[date_col_social], format='%b-%y', errors='coerce'
    )
    if df_social['parsed_date'].isna().all():
        df_social['parsed_date'] = pd.to_datetime(
            df_social[date_col_social], format='%m/%Y', errors='coerce'
        )

    # Isolate the training window spanning from July 2011 up to December 2023
    end_of_training_window = pd.to_datetime("2023-12-01")

    train_hackmageddon = df_hackmageddon[
        df_hackmageddon['parsed_date'] <= end_of_training_window
    ].copy()
    train_social = df_social[
        df_social['parsed_date'] <= end_of_training_window
    ].copy()

    # Drop the temporary parsed date column prior to saving the files
    train_hackmageddon.drop(columns=['parsed_date'], inplace=True)
    train_social.drop(columns=['parsed_date'], inplace=True)

    # Export the sliced training files for the univariate and multivariate models
    train_hackmageddon.to_csv(OUTPUT_HACK_TRAIN, index=False)
    train_social.to_csv(OUTPUT_V2_TRAIN, index=False)

    print(f"Exported {len(train_hackmageddon)} rows to {OUTPUT_HACK_TRAIN.name}")
    print(f"Exported {len(train_social)} rows to {OUTPUT_V2_TRAIN.name}")
    print("Batch 1 complete. Data is ready for the 2024 holdout validation.")


if __name__ == "__main__":
    verify_and_slice_data()