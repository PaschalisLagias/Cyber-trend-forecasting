# Config/paths.py
import os
from pathlib import Path

# --- Dynamic Experiment Tag ---
# Defaults to empty string. To use a tag, set it in your notebook before running scripts.
# Example: os.environ['EXPERIMENT_TAG'] = "_Mark3"
TAG = os.environ.get('EXPERIMENT_TAG', '')

# --- Project Root ---
# Dynamically resolves the root directory (Cyber-trend-forecasting)
# Assumes this script is located in Cyber-trend-forecasting/Config/paths.py
ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Data Preparation Directory & Files ---
DATA_PREP_DIR = ROOT_DIR / "Data_Preparation"

RAW_DATA_CSV = DATA_PREP_DIR / "Cyber_Trend_Forecasting.csv"
RAW_DATA_ALL_CSV = DATA_PREP_DIR / "Cyber_Trend_Forecasting_All.csv"

NORM_DATA_CSV = DATA_PREP_DIR / "Norm_CyberTrend_Forecasting.csv"
NORM_DATA_ALL_CSV = DATA_PREP_DIR / "Norm_CyberTrend_Forecasting_All.csv"

SMOOTHED_DATA_CSV = DATA_PREP_DIR / "Smoothed_CyberTrend_Forecasting.csv"
SMOOTHED_DATA_ALL_CSV = DATA_PREP_DIR / "Smoothed_CyberTrend_Forecasting_All.csv"

DATA_PREP_SCRIPT = DATA_PREP_DIR / "data_preparation.py"
DATA_PREP_README = DATA_PREP_DIR / "README.MD"
DATA_PREP_REQS = DATA_PREP_DIR / "requirements.txt"

HACKMAGEDDON_DIR = DATA_PREP_DIR / "Hackmageddon_Attacks"
HACK_NOI_MONTHLY_V2 = HACKMAGEDDON_DIR / "NoI_monthly_v2.csv"

V2_1_DATA_CSV = DATA_PREP_DIR / "Cyber_Trend_Forecasting_v2_1.csv"
TRAIN_HACK_CSV = DATA_PREP_DIR / "train_hackmageddon_2011_2023.csv"
TRAIN_V2_1_CSV = DATA_PREP_DIR / "train_v2_1_2011_2023.csv"
STRATIFIED_VALIDATION_CSV = DATA_PREP_DIR / "stratified_mean_2024_validation.csv"

# Mark 3 SARIMAX Dataset
V2_2_SARIMAX_CSV = DATA_PREP_DIR / "Cyber_Trend_Forecasting_All_v2_2_sarimax.csv"

# --- B-MTGNN Directory & Files ---
BMTGNN_DIR = ROOT_DIR / "B-MTGNN"
BMTGNN_DATA_DIR = BMTGNN_DIR / "data"

BMTGNN_DATA_CSV = BMTGNN_DATA_DIR / "data.csv"
BMTGNN_DATA_TXT = BMTGNN_DATA_DIR / "data.txt"
BMTGNN_GRAPH_CSV = BMTGNN_DATA_DIR / "graph.csv"

BMTGNN_SM_DATA_CSV = BMTGNN_DATA_DIR / "sm_data.csv"
BMTGNN_SM_DATA_G_CSV = BMTGNN_DATA_DIR / "sm_data_g.csv"
BMTGNN_SM_DATA_G_TXT = BMTGNN_DATA_DIR / "sm_data_g.txt"

# --- Processed Data & Comparison Plots ---
PROCESSED_DATA_DIR = ROOT_DIR / "Processed_Data"
COMPARISON_PLOTS_DIR = PROCESSED_DATA_DIR / "Comparison_Plots"

# B-MTGNN Processed Data Outputs & Archives
PROCESSED_BMTGNN_DIR = PROCESSED_DATA_DIR / "B-MTGNN"
BMTGNN_ARCHIVE_DIR = PROCESSED_BMTGNN_DIR / "Archive"
BMTGNN_WORKING_TXT = PROCESSED_BMTGNN_DIR / "sm_data.txt"

# VisionTS Processed Data Outputs & Archives
PROCESSED_VISION_DIR = PROCESSED_DATA_DIR / "VisionTS"
VISION_ARCHIVE_DIR = PROCESSED_VISION_DIR / "Archive"
VISION_WORKING_CSV = PROCESSED_VISION_DIR / "Mark3_Clipped_Data.csv"

# Output paths dynamically inject the tag
BMTGNN_PREDICTIONS = PROCESSED_BMTGNN_DIR / f"predictions{TAG}.npy" 
BMTGNN_CONFIDENCE = PROCESSED_BMTGNN_DIR / f"confidence{TAG}.npy" 
BMTGNN_HISTORY = PROCESSED_BMTGNN_DIR / f"history_data{TAG}.npy" 
BMTGNN_NAMES = PROCESSED_BMTGNN_DIR / f"node_names{TAG}.npy"

# Legacy Archive Reference
LEGACY_MARK2_TXT = PROCESSED_DATA_DIR / 'Legacy_Archives' / 'mark2_unsmoothed.txt'

# B-MTGNN Comparison Subdirectories
BMTGNN_GLOBAL_DIR = COMPARISON_PLOTS_DIR / "BMTGNN_Global"
BMTGNN_GAPS_DIR = COMPARISON_PLOTS_DIR / "BMTGNN_Gaps"
BMTGNN_FORECAST_DIR = COMPARISON_PLOTS_DIR / "BMTGNN_Forecast"
BMTGNN_OUTLOOK_DIR = COMPARISON_PLOTS_DIR / "BMTGNN_Outlook"

# VisionPP Comparison Subdirectories
VISION_GLOBAL_DIR = COMPARISON_PLOTS_DIR / "VisionPP_Global"
VISION_GAPS_DIR = COMPARISON_PLOTS_DIR / "VisionPP_Gaps"
VISION_FORECAST_DIR = COMPARISON_PLOTS_DIR / "VisionPP_Forecast"
VISION_OUTLOOK_DIR = COMPARISON_PLOTS_DIR / "VisionPP_Outlook"

# Comparative Evaluation Table Path
TABLE_9_IMG = COMPARISON_PLOTS_DIR / "Overall" / "Table_9_Comparative_Evaluation.png"