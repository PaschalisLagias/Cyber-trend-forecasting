"""
Unified Comparative Visualisation Pipeline

This script ingests array data from multiple models (e.g., B-MTGNN and VisionTS++)
to generate stylistically consistent Figure 4 visualisations. It replicates the 
aesthetic properties of the original project paper by applying independent max-scaling, 
Seaborn-inspired grid themes, and uncluttered shading logic.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime

# --- Master Translation Dictionaries ---
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

RUN_TIMESTAMP = datetime.now().strftime("%b%d_%H%M%S")

# --- Helper Functions ---

def normalise_series(series):
    """Normalises a series using independent Max-Scaling."""
    mx = np.max(np.abs(series))
    if mx == 0:
        return series
    return series / mx

def exponential_smoothing_clamped(series, alpha=0.01):
    """Applies exponential smoothing, clamped at 0."""
    if len(series) == 0:
        return series
    
    result = [max(0, series[0])]
    for n in range(1, len(series)):
        val = alpha * series[n] + (1 - alpha) * result[n - 1]
        result.append(max(0, val))
        
    return np.array(result)

def clean_string(s):
    """Sanitises strings for legend display."""
    s = str(s).lower().strip()
    s = s.replace('solution_', '').replace('_papers', '').replace('_mentions', '')
    s = s.replace('mentions-', '').replace('-all', '')
    s = s.replace('_', ' ')
    return s.title()

def get_full_pat_name(abbrev):
    """Maps abbreviations to full names."""
    return ABBREVIATION_MAP.get(abbrev, abbrev)

def find_col_index(target_name, all_names):
    """Locates the column index using fuzzy matching."""
    clean_target = clean_string(target_name).lower()
    
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
    """Generates continuous datetime objects."""
    dates = []
    current = start_date
    for _ in range(num_steps):
        dates.append(current)
        year = current.year + (current.month // 12)
        month = (current.month % 12) + 1
        current = current.replace(year=year, month=month, day=1)
    return dates

def double_save_figure(output_dir, filename, fig=None):
    """Saves the figure to the main directory and an archive."""
    target = fig if fig else plt
    main_path = output_dir / filename
    target.savefig(main_path, dpi=300, bbox_inches='tight')
    
    archive_dir = output_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    name, ext = os.path.splitext(filename)
    archive_path = archive_dir / f"{name}_{RUN_TIMESTAMP}{ext}"
    target.savefig(archive_path, dpi=300, bbox_inches='tight')
    
    if fig:
        plt.close(fig)
    else:
        plt.close()

def load_raw_history_csv(csv_path, col_names):
    """Loads historical data directly from a raw CSV file and maps to expected columns."""
    df = pd.read_csv(csv_path)
    if 'Date' in df.columns or 'month' in df.columns.str.lower():
        df = df.select_dtypes(include=[np.number])
        
    full_history = np.zeros((len(df), len(col_names)))
    df_cols_clean = [clean_string(c).lower() for c in df.columns]
    
    for i, target_node in enumerate(col_names):
        clean_node = clean_string(target_node).lower()
        match_idx = -1
        for j, csv_col in enumerate(df_cols_clean):
            if clean_node == csv_col or clean_node in csv_col:
                match_idx = j
                break
        if match_idx != -1:
            full_history[:, i] = df.iloc[:, match_idx].values
            
    return full_history

# --- Plotting Engine ---

def plot_unified_figure_4(model_name, threat_key, pats, preds, trues, cols, dates, f_start, out_dir, conf=None, alpha=0.01):
    """
    Generates a Figure 4 plot matching the original paper aesthetic.
    Applies styling, independent scaling, full-timeline smoothing, and removes PAT stacking.
    """
    t_idx = find_col_index(threat_key, cols)
    if t_idx is None:
        return False 

    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        try:
            plt.style.use('seaborn-darkgrid')
        except OSError:
            pass

    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = [
        "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", 
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#1f77b4"
    ]
    
    f_dates = dates[f_start:]
    
    # 1. Process Threat
    full_t = np.concatenate([trues[:, t_idx], preds[:, t_idx]])
    sm_t = exponential_smoothing_clamped(normalise_series(full_t), alpha=alpha)
    ax.plot(dates, sm_t, color='#1f4e99', linewidth=2.5, label=f"Threat: {clean_string(threat_key)}")
    
    # 2. Process Confidence Interval
    if conf is not None:
        t_max = np.max(np.abs(full_t)) if np.max(np.abs(full_t)) != 0 else 1.0
        c_val = exponential_smoothing_clamped(conf[:, t_idx] / t_max, alpha=alpha)
        
        t_fore_sm = sm_t[f_start:]
        lim = min(len(f_dates), len(c_val))
        ax.fill_between(f_dates[:lim], t_fore_sm[:lim] - c_val[:lim], t_fore_sm[:lim] + c_val[:lim], 
                        color='#1f4e99', alpha=0.2, label='95% Confidence Interval')
    
    # 3. Process Solutions
    c_idx = 0
    valid_pats = []
    
    for pat_abbrev in pats:
        full_pat_name = get_full_pat_name(pat_abbrev)
        s_idx = find_col_index(full_pat_name, cols)
        if s_idx is not None:
            valid_pats.append((s_idx, full_pat_name))
            
    for s_idx, sol_name in valid_pats:
        full_s = np.concatenate([trues[:, s_idx], preds[:, s_idx]])
        sm_s = exponential_smoothing_clamped(normalise_series(full_s), alpha=alpha)
        
        c = colors[c_idx % len(colors)]
        ax.plot(dates, sm_s, color=c, linewidth=1.5, label=clean_string(sol_name))
        c_idx += 1

    # 4. Styling & Unified Legend
    ax.set_title(f"Figure 4: {clean_string(threat_key)} ({model_name} Outlook)", fontsize=16, pad=10)
    ax.set_xlabel('Timeline', fontsize=12)
    ax.set_ylabel('Trend', fontsize=12)
    ax.set_ylim(bottom=0)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    
    ax.axvspan(dates[f_start], dates[-1], color='#f5c6cb', alpha=0.4, label='Forecast Window')
    
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 5:
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    else:
        ax.legend(handles, labels, loc='upper left')

    file_name = f"Fig4_{model_name.replace(' ', '')}_{clean_string(threat_key).replace(' ', '_')}.png"
    double_save_figure(out_dir, file_name, fig=fig)
    return True

# --- Main Entry Interface ---

def generate_model_comparisons(model_name, preds_path, cols_path, out_dir, trues_path=None, history_csv_path=None, conf_path=None, alpha=0.01):
    """
    Ingests array paths for a specified model and generates comparative Figure 4 outputs.
    Can load column names from either .npy arrays or .pkl metadata files.
    """
    print("=" * 60)
    print(f"--- Generating Visualisations for {model_name} (alpha={alpha}) ---")
    
    try:
        preds = np.load(preds_path).squeeze()
        if preds.ndim == 3:
            preds = preds[-1, :, :]
            
        # Dynamically load column names
        cols_file = Path(cols_path)
        if cols_file.suffix == '.pkl':
            with open(cols_file, 'rb') as f:
                metadata = pickle.load(f)
            cols = metadata.get('selected_feature_names', metadata.get('feature_names', []))
            cols = list(cols) if isinstance(cols, (np.ndarray, list)) else []
        else:
            cols = np.load(cols_file, allow_pickle=True).tolist()
            
        # Determine history loading method
        csv_file = Path(str(history_csv_path)) if history_csv_path else None
        if csv_file and csv_file.exists():
            print(f"Loading history from CSV: {csv_file.name}")
            trues = load_raw_history_csv(csv_file, cols)
        else:
            if csv_file:
                print(f"Warning: CSV not found at {csv_file}. Falling back to array.")
            trues = np.load(trues_path).squeeze()
            if trues.ndim == 3:
                trues = trues[-1, :, :]
                
        conf = np.load(conf_path).squeeze() if conf_path and Path(conf_path).exists() else None
        print(f"Data loaded successfully for {model_name}. (Features: {len(cols)})")
    except Exception as e:
        print(f"Failed to load data for {model_name}: {e}")
        return

    f_start = trues.shape[0] 
    f_horizon = preds.shape[0] 
    dates = generate_date_labels_forward(datetime(2011, 7, 1), f_start + f_horizon)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    b_count = 0
    for threat, pats in THREAT_PAT_MAP.items():
        if plot_unified_figure_4(model_name, threat, pats, preds, trues, cols, dates, f_start, out_path, conf, alpha):
            b_count += 1

    print(f"SUCCESS: Generated {b_count} plots in {out_path.name}/")
    print("=" * 60)