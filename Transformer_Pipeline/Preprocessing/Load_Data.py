import os

import pandas as pd


def load_cyber_threat_data(file_path: str = '../../Data_Preparation/Cyber_Trend_Forecasting_All.csv') -> pd.DataFrame | None:
    """Loads the cyber threat time-series data from the specified CSV file."""
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, file_path)

    print(f"DEBUG: Attempting to load file from absolute path: {abs_file_path}")
    if not os.path.exists(abs_file_path):
        print(f"Error: File not found at {abs_file_path}")
        return None

    try:
        print(f"Loading data from {abs_file_path}...")
        df = pd.read_csv(abs_file_path, parse_dates=['Date'], index_col='Date')
        df.sort_index(inplace=True)
        print("Data loaded successfully.")
        print(f"Shape of the DataFrame: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nLast 5 rows:")
        print(df.tail())
        return df
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

if __name__ == '__main__':
    data = load_cyber_threat_data()

    if data is not None:
        print("\nData summary:")
        print(data.info())