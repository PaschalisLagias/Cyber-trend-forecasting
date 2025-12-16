import torch
import subprocess
import sys
import os

def setup_environment():
    """
    Installs requirements and checks GPU status.
    Looks for requirements.txt in the Current Working Directory.
    """
    print(f"\n--- Initialising Environment (CWD: {os.getcwd()}) ---")
    
    # 1. Install Requirements
    req_file = "requirements.txt"
    
    if os.path.exists(req_file):
        print(f"Found {req_file}. Installing dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            print("Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
    else:
        print(f"WARNING: {req_file} not found in current directory.")
        print("Ensure you have set os.chdir() to the 'Transformer_Pipeline' folder.")

    # 2. Check GPU
    if torch.cuda.is_available():
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected. Go to Runtime > Change runtime type > T4 GPU")