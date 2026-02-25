# Config/paths.py
import os
from pathlib import Path

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

# --- B-MTGNN Directory & Files ---
BMTGNN_DIR = ROOT_DIR / "B-MTGNN"
BMTGNN_DATA_DIR = BMTGNN_DIR / "data"

BMTGNN_DATA_CSV = BMTGNN_DATA_DIR / "data.csv"
BMTGNN_DATA_TXT = BMTGNN_DATA_DIR / "data.txt"
BMTGNN_GRAPH_CSV = BMTGNN_DATA_DIR / "graph.csv"

BMTGNN_SM_DATA_CSV = BMTGNN_DATA_DIR / "sm_data.csv"
BMTGNN_SM_DATA_TXT = BMTGNN_DATA_DIR / "sm_data.txt"
BMTGNN_SM_DATA_G_CSV = BMTGNN_DATA_DIR / "sm_data_g.csv"
BMTGNN_SM_DATA_G_TXT = BMTGNN_DATA_DIR / "sm_data_g.txt"

# --- Processed Data & Comparison Plots ---
PROCESSED_DATA_DIR = ROOT_DIR / "Processed_Data"
COMPARISON_PLOTS_DIR = PROCESSED_DATA_DIR / "Comparison_Plots"

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