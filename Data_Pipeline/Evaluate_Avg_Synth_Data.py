"""
Stratified Monthly Mean Holdout Validation
==========================================

This script calculates the historical monthly average for every cyberattack
vector across the 2011 to 2023 training window. It then projects these
seasonal averages onto the 2024 calendar year and calculates the Mean
Absolute Error against the actual 2024 Hackmageddon incident counts.
The resulting error score provides a benchmark to compare against
multivariate time-series models.
"""

import sys
from pathlib import Path
import pandas as pd

# Import paths dynamically from the central Config file
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Config.Paths import (
    TRAIN_HACK_CSV as TRAIN_DATA_PATH,
    HACK_NOI_MONTHLY_V2 as FULL_HACKMAGEDDON_PATH,
    STRATIFIED_VALIDATION_CSV as OUTPUT_COMPARISON_PATH
)

def parse_date_column(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Standardises date strings into Pandas datetime objects, handling
    both MM/YYYY and MMM-YY formatting variations seamlessly.
    """
    parsed = pd.to_datetime(df[col_name], format='%m/%Y', errors='coerce')
    if parsed.isna().all():
        parsed = pd.to_datetime(df[col_name], format='%b-%y', errors='coerce')
    return parsed


def run_validation():
    """
    Executes the calculation of monthly means, projects them across 2024,
    and evaluates the prediction error against real world incident counts.
    """
    print("Loading 2011-2023 training data and full incident history...")
    if not TRAIN_DATA_PATH.exists() or not FULL_HACKMAGEDDON_PATH.exists():
        print("ERROR: Could not find required input CSV files.")
        sys.exit(1)

    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_full = pd.read_csv(FULL_HACKMAGEDDON_PATH)

    # Identify the date column and all numerical attack categories
    date_col_train = df_train.columns[0]
    date_col_full = df_full.columns[0]
    attack_columns = [col for col in df_train.columns if col != date_col_train]

    # Convert date strings to datetime objects for temporal indexing
    df_train['datetime'] = parse_date_column(df_train, date_col_train)
    df_full['datetime'] = parse_date_column(df_full, date_col_full)

    # Extract the calendar month number to group historical seasonality
    df_train['month_num'] = df_train['datetime'].dt.month

    # Calculate the average attack count for each calendar month across 2011-2023
    print("Calculating historical stratified monthly means...")
    monthly_averages = df_train.groupby('month_num')[attack_columns].mean()

    # Isolate the actual ground-truth incident records for the 2024 calendar year
    start_2024 = pd.to_datetime("2024-01-01")
    end_2024 = pd.to_datetime("2024-12-31")
    
    ground_truth_2024 = df_full[
        (df_full['datetime'] >= start_2024) & (df_full['datetime'] <= end_2024)
    ].copy()

    if ground_truth_2024.empty:
        print("ERROR: No 2024 records found in the full Hackmageddon file.")
        sys.exit(1)

    # Project the historical monthly averages onto the 2024 timeline
    ground_truth_2024['month_num'] = ground_truth_2024['datetime'].dt.month
    predictions_2024 = ground_truth_2024[['datetime', 'month_num']].copy()

    # Merge the calculated averages based on the corresponding calendar month
    predictions_2024 = predictions_2024.merge(
        monthly_averages, on='month_num', how='left'
    )

    # Calculate the Mean Absolute Error across all attack columns for 2024
    print("Evaluating error against actual 2024 Hackmageddon records...")
    actual_matrix = ground_truth_2024[attack_columns].values
    predicted_matrix = predictions_2024[attack_columns].values

    absolute_errors = pd.DataFrame(
        abs(actual_matrix - predicted_matrix), columns=attack_columns
    )
    overall_mae = absolute_errors.values.mean()

    # Export a side-by-side comparison file for human verification
    comparison_export = pd.DataFrame({
        'Date': ground_truth_2024[date_col_full].values,
        'Actual_Total_Incidents': ground_truth_2024[attack_columns].sum(axis=1).values,
        'Predicted_Total_Incidents': predictions_2024[attack_columns].sum(axis=1).values,
        'Monthly_MAE': absolute_errors.mean(axis=1).values
    })
    comparison_export.to_csv(OUTPUT_COMPARISON_PATH, index=False)

    print("\n" + "=" * 50)
    print("2024 Holdout Validation Results (Stratified Mean)")
    print("=" * 50)
    print(f"Overall Mean Absolute Error (MAE): {overall_mae:.4f}")
    print(f"Comparison report saved to: {OUTPUT_COMPARISON_PATH.name}")
    print("=" * 50)
    print("Ready to compare vs predictive scores.")


if __name__ == "__main__":
    run_validation()