import pandas as pd
import os

def load_cyber_threat_data(file_path='../../Data_Preparation/Cyber_Trend_Forecasting_All.csv'):
    """
    Loads the cyber threat time-series data from the specified CSV file.

    Args:
        file_path (str): The relative path to the CSV file from the projects's location.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data, with the 'Date' column
                      parsed and set as the index. Returns None if the file is not found.
    """
    # Construct the absolute path based on the script's location
    script_dir = os.path.dirname(__file__) # Gets working dir
    abs_file_path = os.path.join(script_dir, file_path)

    # Check PATH (incase of error)
    print(f"DEBUG: Attempting to load file from absolute path: {abs_file_path}")
    if not os.path.exists(abs_file_path):
        print(f"Error: File not found at {abs_file_path}")
        return None

    try:
        print(f"Loading data from {abs_file_path}...")
        # Read the CSV, parse the 'Date' column, and set it as the index
        df = pd.read_csv(abs_file_path, parse_dates=['Date'], index_col='Date')
        df.sort_index(inplace=True) # Ensure data is chronologically sorted
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
    # Run the script directly to test the loading function
    # Will include this in addition to Vignette
    data = load_cyber_threat_data()

    if data is not None:
        print("\nData summary:")
        print(data.info())