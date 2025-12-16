import os
import pandas as pd


def load_cyber_threat_data(
    file_path: str = "../../Data_Preparation/Cyber_Trend_Forecasting_All.csv",
):  # -> pd.DataFrame | None:
    """
    Loads and preprocesses cyber threat time-series data from a CSV file.

    This function resolves the file path relative to the script's location,
    parses the 'Date' column as the datetime index, and sorts the data chronologically.

    Args:
        file_path (str, optional): The relative path to the CSV file.
            Defaults to '../../Data_Preparation/Cyber_Trend_Forecasting_All.csv'.

    Returns:
        pd.DataFrame | None: A pandas DataFrame indexed by Date if loading is successful;
            returns None if the file is not found or an error occurs.
    """
    script_dir = os.path.dirname(__file__)

    # UPDATE TO PATH HANDLING
    # Handle both absolute and relative paths safely
    if os.path.isabs(file_path):
        abs_file_path = file_path
    else:
        abs_file_path = os.path.join(script_dir, file_path)

    print(f"DEBUG: Attempting to load file from: {abs_file_path}")
    if not os.path.exists(abs_file_path):
        print(f"Error: File not found at {abs_file_path}")
        return None

    try:
        # Read CSV without parsing dates initially
        print(f"Loading raw data...")
        df = pd.read_csv(abs_file_path)

        # Explicitly parse the 'Date' column
        print("Parsing dates with format '%b-%y'...")
        df["Date"] = pd.to_datetime(df["Date"], format="%b-%y")

        # Set Index and Sort Chronologically
        df.set_index("Date", inplace=True)
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
