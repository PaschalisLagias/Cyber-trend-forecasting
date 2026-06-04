"""
Comparative Visualisation Pipeline
==================================================

TODO: This script is currently designed to generate independent Figure 4 
comparison plots for the B-MTGNN and VisionPP models based on the specified 
NATO deliverables. In future iterations, this will be refactored into a 
general-purpose visualisation script capable of ingesting N-models dynamically.

ARCHITECTURAL NOTE (Plot-Specific Normalisation vs Categorical Global):
    The original script used "Categorical Global Normalisation," scaling all 
    threats by the single highest threat in the entire dataset. In the new dataset, 
    massive outliers (e.g., DDoS) squashed smaller threats (e.g., Malware) into 
    unreadable flat lines near 0.0. 
    
    This script implements "Plot-Specific Normalisation." For each Figure 4 plot, 
    the script calculates the absolute maximum value exclusively among the Target 
    Threat and its Pertinent Alleviation Technologies (PATs). It then scales 
    all lines in that specific plot to 1.0 using this shared local maximum. 
    This perfectly preserves the mathematical gap (Trend vs PAT) required for 
    the filtering rule while ensuring every chart is organically scaled and readable.

Usage:
    Run from the Data_Pipeline directory:
    $ python Visualise_Comparison.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime


# --- Configuration & Paths ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Config.Paths import *

# Output Paths inherited from Config
BMTGNN_OUT = BMTGNN_OUTLOOK_DIR
VISION_OUT = VISION_OUTLOOK_DIR

# Ensure output directories exist
for directory in [BMTGNN_OUT, VISION_OUT]:
    directory.mkdir(parents=True, exist_ok=True)

# Global timestamp for the Double Save Archive
RUN_TIMESTAMP = datetime.now().strftime("%b%d_%H%M%S")

# --- Hardcoded Target Definitions ---
THREAT_PAT_MAP = {
    "Account Hijacking": ["AC", "AD", "CAPTCHA", "CR", "IDS/IPS", "IdM", "LP", "MFA", "ML/DL", "NLP/LLM", "PT", "SM"],
    "Adversarial Attack": ["AD", "AdT", "BN", "DA", "DD", "DP", "DR", "DS", "ML/DL", "NI", "NLP/LLM", "OD", "RRAM", "SS", "TAI"],
    "APT": ["AC", "DLP", "DRM", "DT", "GT", "IDS/IPS", "LP", "MFA", "ML/DL", "NLP/LLM", "NS", "PT", "RA", "UBA"],
    "Backdoor": ["AD", "DAS", "IDS/IPS", "ML/DL", "PT", "SA"],
    "Botnet": ["AD", "BC", "BH", "BT", "CAPTCHA", "GM", "GT", "HP", "IDS/IPS", "ML/DL", "NLP/LLM", "PF", "PT", "RC", "RL", "SDN", "TS"],
    "Brute Force Attack": ["CAPTCHA", "CR", "DBI", "IDS/IPS", "MFA", "ML/DL", "OTP", "PH", "PT"],
    "Cryptojacking": ["BT", "ML/DL", "PT", "TA"],
    "DDoS": ["BC", "BH", "BT", "IDS/IPS", "ML/DL", "NLP/LLM", "PF", "PT", "RC", "RL", "TS"],
    "Data Poisoning": ["AD", "AdT", "BN", "DP", "DS", "ML/DL", "NLP/LLM", "OD", "TAI"],
    "Deepfake": ["3DFR", "AD", "BO", "DW", "LD", "ML/DL", "NLP/LLM"],
    "Disinformation": ["BC", "CA", "DLT", "DP", "DT", "GT", "HG", "IR", "ML/DL", "NLP/LLM", "SI"],
    "DNS Spoofing": ["BC", "CR", "DNSSEC", "ML/DL", "PT", "RA"],
    "Dropper": ["AW", "CS", "FIM", "IDS/IPS", "ML/DL", "NLP/LLM", "PT", "SBX"],
    "Insider Threat": ["AC", "AD", "AM", "AT", "CR", "DLD", "IDS/IPS", "KD", "LP", "ML/DL", "MTD", "NLP/LLM", "PT", "UBA"],
    "IoT Device Attack": ["AD", "BC", "CR", "IDS/IPS", "IdM", "MFA", "ML/DL", "MS", "PT", "SB"],
    "Malware": ["AC", "AD", "AW", "BBD", "BC", "CR", "CS", "DAS", "DB", "DM", "DT", "FIM", "FV", "GT", "HP", "IDS/IPS", "ML/DL", "NLP/LLM", "PMT", "PT", "SA", "SB", "SBX", "SHMM", "SMF", "VK"],
    "MITM": ["BC", "CAPTCHA", "CP", "CR", "ML/DL", "PKI", "PT", "SSL/TLS", "SSP", "VPN"],
    "Password Attack": ["CAPTCHA", "CR", "GA", "IDS/IPS", "MA", "MFA", "ML/DL", "NLP/LLM", "OTP", "PH", "PM", "PP", "PSM", "PT"],
    "Phishing": ["AC", "BT", "CR", "DT", "MA", "MFA", "ML/DL", "NLP/LLM", "PKI"],
    "Ransomware": ["AC", "AD", "AW", "BC", "CR", "DAS", "DB", "DT", "IDS/IPS", "ML/DL", "NLP/LLM", "PMT", "PT", "SA", "SHMM"],
    "Session Hijacking": ["AD", "CA", "CR", "Https", "IBE", "ML/DL", "PT", "SAT", "SM", "SSL/TLS"],
    "Supply Chain Attack": ["AC", "AD", "BC", "CR", "IdM", "ML/DL", "NLP/LLM", "PT", "SCRM"],
    "Targeted Attack": ["AC", "DRM", "DT", "GT", "IDS/IPS", "LP", "MFA", "ML/DL", "NLP/LLM", "NS", "PT", "RA", "UBA"],
    "Trojan": ["AD", "BBD", "CR", "FV", "GT", "IDS/IPS", "ML/DL", "NLP/LLM", "PT", "SMF"],
    "Vulnerability": ["CFI", "IDS/IPS", "ML/DL", "NLP/LLM", "PMT", "PT", "SC", "SIEM", "VA", "VM", "VS"],
    "Zero-day": ["AD", "DT", "FIM", "GT", "IDS/IPS", "ML/DL", "NLP/LLM", "PrP", "VM", "VPN"]
}

# Master translation dictionary for abbreviations
ABBREVIATION_MAP = {
    "AC": "Access Control", "AD": "Anomaly Detection", "CAPTCHA": "Captcha", "CR": "Cryptography",
    "IDS/IPS": "IDS/IPS", "IdM": "Identity Management", "LP": "Least Privilege", "MFA": "Multi Factor Authentication",
    "ML/DL": "Machine Learning", "NLP/LLM": "NLP/LLM", "PT": "Penetration Testing", "SM": "Session Management",
    "AdT": "Adversarial Training", "BN": "Bayesian Network", "DA": "Data Augmentation", "DD": "Defensive Distillation",
    "DP": "Data Provenance", "DR": "Dimensionality Reduction", "DS": "Data Sanitization", "NI": "Noise Injection",
    "OD": "Outlier Detection", "RRAM": "RRAM", "SS": "Spatial Smoothing", "TAI": "Trustworthy AI",
    "DLP": "Data Loss Prevention", "DRM": "Dynamic Resource Management", "DT": "Deception Technology", "GT": "Game Theory",
    "NS": "Network Segmentation", "RA": "Risk Assessment", "UBA": "User Behavior Analytics", "DAS": "Dynamic Analysis",
    "SA": "Static Analysis", "BC": "Blockchain", "BH": "Blackholing", "BT": "Biometrics", "GM": "Graphical Model",
    "HP": "Honeypot", "PF": "Packet Filtering", "RC": "Rank Correlation", "RL": "Rate Limiting",
    "SDN": "Software Defined Network", "TS": "Traffic Shaping", "DBI": "Dynamic Binary Instrumentation",
    "OTP": "One Time Password", "PH": "Password Hashing", "TA": "Taint Analysis", "3DFR": "3D Face Reconstruction",
    "BO": "Biometrics", "DW": "Digital Watermark", "LD": "Liveness Detection", "CA": "Continuous Authentication",
    "DLT": "Distributed Ledgers", "HG": "Hypergame", "IR": "Image Recognition", "SI": "Source Identification",
    "DNSSEC": "DNSSEC", "AW": "Application Whitelisting", "CS": "Code Signing", "FIM": "File Integrity Monitoring",
    "SBX": "Sandboxing", "AM": "Activity Monitoring", "AT": "Attack Tree", "DLD": "Data Leakage Detection",
    "KD": "Keystroke Dynamics", "MTD": "Moving Target Defense", "MS": "Mutual Authentication", "SB": "Secure Boot",
    "BBD": "Behavior Based Detection", "DB": "Data Backups", "DM": "Darknet Monitoring", "FV": "Formal Verification",
    "PMT": "Patch Management", "SHMM": "Hidden Markov Model", "SMF": "Split Manufacturing", "VK": "Virtual Keyboards",
    "CP": "Certificate Pinning", "PKI": "Public Key Infrastructure", "SSL/TLS": "SSL/TLS", "SSP": "Secure Simple Pairing",
    "VPN": "VPN", "GA": "Graphical Authentication", "MA": "Mutual Authentication", "PM": "Password Management",
    "PP": "Password Policy", "PSM": "Password Strength Meters", "Https": "HTTPS", "IBE": "Identity-Based Encryption",
    "SAT": "Strong Authentication", "SCRM": "Supply Chain Risk Management", "CFI": "Control Flow Integrity",
    "SC": "Standardized Communication", "SIEM": "SIEM", "VA": "Vulnerability Assessment", "VM": "Vulnerability Management",
    "VS": "Vulnerability Scanner", "PrP": "Privacy Preserving"
}

# --- Helper Functions ---

def exponential_smoothing(series, alpha=0.1):
    """
    Applies exponential smoothing to reduce noise, explicitly clamping at 0
    to prevent illogical negative dips in the graph.
    """
    if len(series) == 0:
        return series
    
    result = [max(0, series[0])]
    for n in range(1, len(series)):
        val = alpha * series[n] + (1 - alpha) * result[n - 1]
        result.append(max(0, val))
        
    return np.array(result)

def clean_string(s):
    """Sanitises feature strings for clean legend display."""
    s = str(s).lower().strip()
    s = s.replace('solution_', '').replace('_papers', '').replace('_mentions', '')
    s = s.replace('mentions-', '').replace('-all', '')
    s = s.replace('_', ' ')
    return s.title()

def get_full_pat_name(abbrev):
    """Maps an abbreviation to its full name."""
    return ABBREVIATION_MAP.get(abbrev, abbrev)

def find_col_index(target_name, all_names):
    """Fuzzy matching to link a requested name to the correct array index."""
    clean_target = clean_string(target_name).lower()
    
    # VisionPP Model Exception Handling
    if clean_target == "apt":
        clean_target = "advanced persistent threat"
        
    for i, raw_name in enumerate(all_names):
        clean_col = clean_string(raw_name).lower()
        if (clean_target == clean_col or 
            clean_target in clean_col or 
            clean_col in clean_target):
            return i
            
    return None

def generate_date_labels_forward(start_date, num_steps):
    """Generates continuous datetime objects for the x-axis."""
    dates = []
    current = start_date
    for _ in range(num_steps):
        dates.append(current)
        year = current.year + (current.month // 12)
        month = (current.month % 12) + 1
        current = current.replace(year=year, month=month, day=1)
    return dates

def double_save_figure(output_dir, filename, show_inline=True, fig=None):
    """Saves the matplotlib figure to the main directory and an archive."""
    target = fig if fig else plt
    main_path = output_dir / filename
    target.savefig(main_path, dpi=300, bbox_inches='tight')
    
    archive_dir = output_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    name, ext = os.path.splitext(filename)
    archive_path = archive_dir / f"{name}_{RUN_TIMESTAMP}{ext}"
    target.savefig(archive_path, dpi=300, bbox_inches='tight')

    if show_inline:
        if fig:
            from IPython.display import display
            display(fig)
        else:
            plt.show()
    
    if fig:
        plt.close(fig)
    else:
        plt.close()
    
    print(f"Saved: {filename} (and archived to {archive_dir.name}/)")

# --- Core Plotting Function ---

def plot_figure_4(model_name, threat_key, pats, preds, trues, conf, cols, dates, f_start, out_dir):
    """
    Generates a Figure 4 style continuous trend plot using Plot-Specific Normalisation.
    The Threat and its specific PATs are scaled based on the maximum volume found 
    only within their isolated subset, preserving relational gaps perfectly.
    """
    t_idx = find_col_index(threat_key, cols)
    if t_idx is None:
        return False # Threat missing from array

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    
    # 1. Identify Plot-Specific Maximum across Threat AND its PATs
    # This prevents an external outlier (like DDoS) from squashing this chart.
    local_data_arrays = [trues[:, t_idx], preds[:, t_idx]]
    
    valid_pat_indices = []
    for pat_abbrev in pats:
        full_pat_name = get_full_pat_name(pat_abbrev)
        s_idx = find_col_index(full_pat_name, cols)
        if s_idx is not None:
            valid_pat_indices.append(s_idx)
            local_data_arrays.extend([trues[:, s_idx], preds[:, s_idx]])
            
    # Calculate the localized peak to use as the denominator
    plot_max = np.max(np.abs(np.concatenate(local_data_arrays)))
    if plot_max == 0:
        plot_max = 1.0
        
    # 2. Process Threat (Scaled by plot_max)
    sm_hist = exponential_smoothing(trues[:, t_idx] / plot_max)
    sm_fore = exponential_smoothing(preds[:, t_idx] / plot_max)
    
    sm_fore_connected = np.insert(sm_fore, 0, sm_hist[-1])
    f_dates = dates[f_start-1:]
    
    ax.plot(dates[:f_start], sm_hist, color='black', linewidth=2.5, label=f"Threat: {clean_string(threat_key)}")
    ax.plot(f_dates, sm_fore_connected, color='black', linewidth=2.5)

    # Threat Confidence Interval
    if conf is not None:
        c_val = exponential_smoothing(conf[:, t_idx] / plot_max)
        c_val_connected = np.insert(c_val, 0, 0) 
        # Plotted without a label here to prevent duplicate legend entries
        ax.fill_between(f_dates, sm_fore_connected - c_val_connected, sm_fore_connected + c_val_connected, 
                        color='mistyrose', alpha=0.5)

    # 3. Process Solutions (Scaled by plot_max to maintain parity)
    c_idx = 0
    for s_idx in valid_pat_indices:
        raw_s_hist = trues[:, s_idx]
        raw_s_fore = preds[:, s_idx]
        
        sm_s_hist = exponential_smoothing(raw_s_hist / plot_max)
        sm_s_fore = exponential_smoothing(raw_s_fore / plot_max)
        sm_s_fore_connected = np.insert(sm_s_fore, 0, sm_s_hist[-1])
        
        # Filtering Rule: Plot only if Solution forecast mean is strictly lower than Threat forecast mean
        if np.mean(sm_s_fore) < np.mean(sm_fore):
            c = colors[c_idx % len(colors)]
            
            # Extract name directly from column index
            pat_name = clean_string(cols[s_idx])
            
            ax.plot(dates[:f_start], sm_s_hist, color=c, linewidth=1.5, label=pat_name)
            ax.plot(f_dates, sm_s_fore_connected, color=c, linewidth=1.5)
            
            # Risk Gap Shading
            ax.fill_between(f_dates, sm_fore_connected, sm_s_fore_connected, 
                            where=(sm_fore_connected > sm_s_fore_connected), color=c, alpha=0.1)

            # Solution Confidence Interval
            if conf is not None:
                c_s_val = exponential_smoothing(conf[:, s_idx] / plot_max)
                c_s_val_connected = np.insert(c_s_val, 0, 0)
                # Plotted without a label here to prevent duplicate legend entries
                ax.fill_between(f_dates, sm_s_fore_connected - c_s_val_connected, sm_s_fore_connected + c_s_val_connected, 
                                color=c, alpha=0.15) 
            c_idx += 1

    # 4. Styling & Unified Legend
    ax.set_title(f"Figure 4: {clean_string(threat_key)} ({model_name} Outlook)", fontsize=16, fontweight='bold')
    ax.set_xlabel('Timeline', fontsize=12)
    ax.set_ylabel('Trend Intensity (Normalised)', fontsize=12)
    
    # Hard lock Y-Axis to prevent illogical negative numbers
    ax.set_ylim(bottom=0)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    ax.axvspan(dates[f_start-1], dates[-1], color='#f0f0f0', alpha=0.5, label='Forecast Horizon')
    
    # Retrieve existing handles and labels
    handles, labels = ax.get_legend_handles_labels()
    
    # Add a unified patch for the 95% Confidence Interval (if B-MTGNN)
    if conf is not None:
        ci_patch = mpatches.Patch(color='mistyrose', alpha=0.7, label='95% Confidence Interval')
        handles.insert(1, ci_patch)
        labels.insert(1, '95% Confidence Interval')
    
    # Format Legend Positioning
    if c_idx > 5:
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    else:
        ax.legend(handles, labels, loc='upper left')

    # Double-Save execution
    file_name = f"Fig4_{clean_string(threat_key).replace(' ', '_')}.png"
    double_save_figure(out_dir, file_name, show_inline=False, fig=fig)
    return True

# --- Main Execution ---

def generate_comparisons():
    print("=" * 60)
    print("--- Initiating Phase 5: Comparative Visualisation ---")
    print("=" * 60)

# 1. Load B-MTGNN Data
    try:
        # Load dynamically tagged arrays using Paths.py variables
        b_preds = np.load(BMTGNN_PREDICTIONS).squeeze() 
        b_conf = np.load(BMTGNN_CONFIDENCE).squeeze()   
        b_hist = np.load(BMTGNN_HISTORY)           
        
        # Load the feature names generated by Forecast_Export (bypassing legacy CSV)
        b_names = np.load(BMTGNN_NAMES, allow_pickle=True).tolist()
        
        print("Successfully loaded B-MTGNN arrays (123 Features).")
    except Exception as e:
        print(f"Failed to load B-MTGNN data: {e}")
        sys.exit(1)

    # # 2. Load VisionPP Data
    # try:
    #     v_preds_raw = np.load(VISION_DIR / 'predictions.npy')
    #     v_preds = v_preds_raw[-1] if v_preds_raw.ndim == 3 else v_preds_raw.squeeze()
    #     v_names = np.load(VISION_DIR / 'feature_names.npy', allow_pickle=True)
    #     print("Successfully loaded VisionPP arrays.")
    # except Exception as e:
    #     print(f"Failed to load VisionPP data: {e}")
    #     sys.exit(1)

    # 3. Set Timeline
    f_start = b_hist.shape[0] 
    f_horizon = b_preds.shape[0] 
    dates = generate_date_labels_forward(datetime(2011, 7, 1), f_start + f_horizon)

    print(f"\nGenerating plots for {len(THREAT_PAT_MAP)} Targets...")
    b_count, v_count = 0, 0

    # 4. Generate Plots
    for threat, pats in THREAT_PAT_MAP.items():
        if plot_figure_4("B-MTGNN", threat, pats, b_preds, b_hist, b_conf, b_names, dates, f_start, BMTGNN_OUT):
            b_count += 1
            
        # if plot_figure_4("VisionTS++", threat, pats, v_preds, b_hist, None, v_names, dates, f_start, VISION_OUT):
        #     v_count += 1

    print("\n" + "=" * 60)
    print("SUCCESS: Comparison Generation Complete!")
    print(f"Generated {b_count} plots for B-MTGNN in: {BMTGNN_OUT.name}/")
    print(f"Generated {v_count} plots for VisionPP in: {VISION_OUT.name}/")
    print("=" * 60)

if __name__ == "__main__":
    generate_comparisons()