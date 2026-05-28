"""
Phase 0: Column Mapping Generator
=================================

This script acts as the translation layer between the legacy B-MTGNN dataset
and the newly scraped dataset (Jan-25). It reads the headers from both files
and applies a set of strict, rule-based transformations to map the legacy 
node names (e.g., 'Mentions-DNS Spoofing') to the new feature names 
(e.g., 'Papers_DNS_Spoofing').

It outputs a 'column_mapping.csv' file that allows human verification of the 
graph node alignment before any data is mathematically processed.

Usage:
    Run this script directly from the Data_Pipeline directory:
    $ python Generate_Column_Mapping.py
"""

import sys
from pathlib import Path
import pandas as pd 

# --- Path Configuration ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Input paths for header extraction
LEGACY_DATA_PATH = PROJECT_ROOT / "B-MTGNN" / "data" / "sm_data_g.csv"
# NEW_DATA_PATH = PROJECT_ROOT / "Data_Preparation" / "Cyber_Trend_Forecasting_All.csv"
# MARK 3 DATA:
NEW_DATA_PATH = PROJECT_ROOT / "Data_Preparation" / "Cyber_Trend_Forecasting_All_v2.csv"

# Output path for the mapping file
OUTPUT_MAPPING_PATH = CURRENT_DIR / "column_mapping.csv"


def translate_header(old_header: str, new_headers_set: set) -> str:
    """
    Applies translation rules to convert a legacy header name to the new format.

    Args:
        old_header (str): The column name from the legacy dataset.
        new_headers_set (set): A set of all available column names in the new dataset.

    Returns:
        str: The matching new header name, or None if no match is found.
    """
    # Rule 1: Exact Match (Handles '-ALL' incident counts and 'Holidays')
    if old_header in new_headers_set:
        return old_header
    
    # Ground Truth Shift (Hackmageddon -> CSIS)
    # Attempts to map legacy Hackmageddon target columns to their CSIS equivalents.
    if "Hackmageddon" in old_header:
        csis_candidate = old_header.replace("Hackmageddon", "CSIS")
        if csis_candidate in new_headers_set:
            return csis_candidate
        
        # Fallback: Check if the new dataset uses a different delimiter for CSIS
        csis_candidate_under = old_header.replace("Hackmageddon", "CSIS").replace("-", "_").replace(" ", "_")
        if csis_candidate_under in new_headers_set:
            return csis_candidate_under

    # Rule 2: Hardcoded Edge Cases (Identified from email forensics)
    exceptions = {
        "WAR/CONFLICT ALL": "War_Conflict_All",
        # Handle specific case formatting changes in the new dataset
        "Solution_Identity-Based Encryption (IBE)_Mentions": "Solution_IDENTITY_BASED_ENCRYPTION_Papers",
        "Solution_Bayesian Network_Mentions": "Solution_BAYESIAN_NETWORK_Papers",
        "Solution_Control Flow Integrity_Mentions": "Solution_CONTROL_FLOW_INTEGRITY_Papers",
        "Solution_MACHINE LEARNING_Mentions": "Solution_ML/DL_Papers"
    }
    if old_header in exceptions and exceptions[old_header] in new_headers_set:
        return exceptions[old_header]

    # Rule 3: Threat Mentions -> Papers (e.g., Mentions-DDoS -> Papers_DDoS)
    if old_header.startswith("Mentions-"):
        threat_name = old_header.replace("Mentions-", "")
        # The new dataset replaces spaces with underscores for these specific columns
        threat_name_under = threat_name.replace(" ", "_")
        new_candidate = f"Papers_{threat_name_under}"
        
        if new_candidate in new_headers_set:
            return new_candidate

    # Rule 4: Solutions (e.g., Solution_ACCESS CONTROL_Mentions -> Solution_ACCESS_CONTROL_Papers)
    if old_header.startswith("Solution_") and old_header.endswith("_Mentions"):
        # Extract the core solution name
        solution_name = old_header.replace("Solution_", "").replace("_Mentions", "")
        # Convert to uppercase and replace spaces/hyphens with underscores
        solution_name_under = solution_name.upper().replace(" ", "_").replace("-", "_")
        new_candidate = f"Solution_{solution_name_under}_Papers"
        
        if new_candidate in new_headers_set:
            return new_candidate

    # If all rules fail, return None to flag for human review
    return None


def generate_mapping():
    """
    Main execution function: Loads headers, translates them, and exports the map.
    """
    print("=" * 60)
    print("--- Initiating Phase 0: Column Mapping Generation ---")
    print("=" * 60)

    # 1. Validate file existence
    for path in [LEGACY_DATA_PATH, NEW_DATA_PATH]:
        if not path.exists():
            print(f"CRITICAL ERROR: Cannot find required file at {path}")
            sys.exit(1)

    # 2. Extract headers (Using nrows=0 to quickly load just the header row)
    print("Extracting headers from both datasets...")
    legacy_df = pd.read_csv(LEGACY_DATA_PATH, nrows=0)
    new_df = pd.read_csv(NEW_DATA_PATH, nrows=0)

    # Filter out temporal columns we don't map to the graph
    legacy_cols = [c for c in legacy_df.columns if 'date' not in c.lower() and 'month' not in c.lower()]
    new_cols_set = set(new_df.columns)

    print(f"Found {len(legacy_cols)} target nodes in legacy dataset.")
    print(f"Found {len(new_cols_set)} available features in new dataset.")

    # 3. Apply translation rules
    mapping_data = []
    missing_cols = []

    for old_col in legacy_cols:
        new_col = translate_header(old_col, new_cols_set)
        
        if new_col:
            mapping_data.append({"Legacy_Node_Name": old_col, "New_Dataset_Column": new_col})
        else:
            missing_cols.append(old_col)

    # 4. Save and report
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv(OUTPUT_MAPPING_PATH, index=False)

    print("\n--- Mapping Results ---")
    print(f"Successfully mapped: {len(mapping_data)} / {len(legacy_cols)} columns.")
    
    if missing_cols:
        print(f"\nWARNING: Failed to map {len(missing_cols)} columns.")
        csis_failures = 0
        for col in missing_cols:
            print(f"  - Missing: {col}")
            if "Hackmageddon" in col or "-ALL" in col:
                csis_failures += 1
        
        if csis_failures > 0:
            print(f"\nAUDIT NOTE: {csis_failures} ground truth/incident nodes failed to map.")
            print("Verify if CSIS covers these specific threat vectors in the Mark 3 dataset.")
    else:
        print("\nSUCCESS: 100% of legacy nodes mapped to the new dataset!")

    print(f"\nMapping file saved for verification at: {OUTPUT_MAPPING_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    generate_mapping()