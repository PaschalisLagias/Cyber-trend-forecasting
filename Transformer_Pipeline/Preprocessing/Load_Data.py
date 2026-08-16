import os
from typing import Optional

import pandas as pd


DATASET_SCHEMAS = {
    "cyber_trend": {
        "date_column": "Date",
        "date_format": "%b-%y",      # e.g. "Jul-11"
        "default_path": "../../Data_Preparation/Cyber_Trend_Forecasting_All.csv",
    },
    "csis": {
        "date_column": "Period",
        "date_format": "%m/%Y",       # e.g. "04/2006"
        "default_path": "../../Data_Preparation/CSIS/csis_output_20260404-01.csv",
    },
    "mark3": {
        "date_column": "Date",
        "date_format": "%b-%y",       # e.g. "Jul-11"
        "default_path": "../../Processed_Data/VisionTS/Mark3_Clipped_Data.csv",
    },
    "v2_1": {
        "date_column": "Date",
        "date_format": "%b-%y",       # e.g. "Jul-11"
        "default_path": "../../Data_Preparation/Cyber_Trend_Forecasting_All_v2_1.csv",
    },
    "v2_2": {
        "date_column": "Date",
        "date_format": "%b-%y",       # e.g. "Jul-11"
        # v2_1 non-attack columns + Hackmageddon attack counts (real through
        # Jan-25, SARIMAX-imputed Feb-Dec 25; see
        # Data_Preparation/Hackmageddon_Attacks/sarimax_gapfill.py)
        "default_path": "../../Data_Preparation/Cyber_Trend_Forecasting_All_v2_2_sarimax.csv",
    },
    # Same raw CSV as v2_2; separate namespace for the all-features (1231) run.
    "v2_2_full": {
        "date_column": "Date",
        "date_format": "%b-%y",
        "default_path": "../../Data_Preparation/Cyber_Trend_Forecasting_All_v2_2_sarimax.csv",
    },
    "v2_2_adaptive": {
        "date_column": "Date",
        "date_format": "%b-%y",       # e.g. "Jul-11"
        # Same raw CSV as v2_2; separate name so the adaptive-selection
        # preprocessing variant gets its own output subdirs.
        "default_path": "../../Data_Preparation/Cyber_Trend_Forecasting_All_v2_2_sarimax.csv",
    },
}


def load_cyber_threat_data(
    file_path: Optional[str] = None,
    dataset: str = "cyber_trend",
):  # -> pd.DataFrame | None:
    """
    Loads and preprocesses cyber threat time-series data from a CSV file.

    Resolves the file path relative to the script's location, parses the
    dataset's date column as the datetime index, and sorts chronologically.

    Args:
        file_path (str, optional): Path to the CSV file. If None, uses the
            default path for the selected dataset.
        dataset (str): Dataset schema to use. One of: 'cyber_trend', 'csis'.
            Determines the date column name and parsing format.

    Returns:
        pd.DataFrame | None: A pandas DataFrame indexed by date if loading is
            successful; None if the file is not found or an error occurs.
    """
    if dataset not in DATASET_SCHEMAS:
        print(f"Error: Unknown dataset '{dataset}'. Known: {list(DATASET_SCHEMAS)}")
        return None

    schema = DATASET_SCHEMAS[dataset]
    date_column = schema["date_column"]
    date_format = schema["date_format"]

    if file_path is None:
        file_path = schema["default_path"]

    script_dir = os.path.dirname(__file__)

    if os.path.isabs(file_path):
        abs_file_path = file_path
    else:
        abs_file_path = os.path.join(script_dir, file_path)

    print(f"DEBUG: Attempting to load file from: {abs_file_path}")
    if not os.path.exists(abs_file_path):
        print(f"Error: File not found at {abs_file_path}")
        return None

    try:
        print(f"Loading raw data (dataset='{dataset}')...")
        df = pd.read_csv(abs_file_path)

        if date_column not in df.columns:
            print(f"Error: Expected date column '{date_column}' not in CSV. Found: {list(df.columns[:5])}...")
            return None

        print(f"Parsing dates from column '{date_column}' with format '{date_format}'...")
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)

        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)

        print("Data loaded and sorted successfully.")
        print(f" - Time Range: {df.index.min().date()} to {df.index.max().date()}")
        print(f" - Total Months: {len(df)}")
        print(f" - DataFrame Shape: {df.shape}")

        return df

    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None


if __name__ == "__main__":
    # Simple test to verify the loader works when run directly
    data = load_cyber_threat_data()
    if data is not None:
        print("\n--- Head of Data ---")
        print(data.head())
        print("\n--- Tail of Data ---")
        print(data.tail())
