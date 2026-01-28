import torch
import subprocess
import sys
import os
import pkg_resources

# def setup_environment():
#     """
#     Installs requirements and checks GPU status.
#     Looks for requirements.txt in the Current Working Directory.
#     """
#     print(f"\n--- Initialising Environment (CWD: {os.getcwd()}) ---")
    
#     # 1. Install Requirements
#     req_file = "requirements.txt"
    
#     if os.path.exists(req_file):
#         print(f"Found {req_file}. Installing dependencies...")
#         try:
#             subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
#             print("Dependencies installed successfully.")
#         except subprocess.CalledProcessError as e:
#             print(f"Error installing dependencies: {e}")
#     else:
#         print(f"WARNING: {req_file} not found in current directory.")
#         print("Ensure you have set os.chdir() to the 'Transformer_Pipeline' folder.")

#     # 2. Check GPU
#     if torch.cuda.is_available():
#         print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
#     else:
#         print("WARNING: No GPU detected. Go to Runtime > Change runtime type > T4 GPU")


def safe_install(target_directory):
    """
    Looks for 'requirements.txt' in the given directory and installs missing packages.
    """
    # Auto-construct the file path
    requirements_file = os.path.join(target_directory, 'requirements.txt')

    print(f"\n--- Checking Dependencies---")
    print(f"Target Directory: {target_directory}")
    
    if not os.path.exists(requirements_file):
        print(f"Error: requirements.txt not found in {target_directory}")
        return

    # 1. Read the requirements file
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    packages_to_install = []
    
    # 2. Check each package
    for req in requirements:
        # Clean the package name (remove version symbols like >=, ==, <)
        # e.g., "numpy<2.0" becomes "numpy"
        pkg_name = req.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()
        
        try:
            # Check if it is already installed
            pkg_resources.get_distribution(pkg_name)
            print(f"{pkg_name} is already installed.")
        except pkg_resources.DistributionNotFound:
            # If not found, add to the list
            print(f"{pkg_name} is MISSING. Queuing for install...")
            packages_to_install.append(req)
        except pkg_resources.VersionConflict:
             print(f"{pkg_name} exists but version might differ. Skipping to avoid conflicts.")

    # 3. Install only the missing ones
    if packages_to_install:
        print(f"\nInstalling {len(packages_to_install)} missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages_to_install)
        print("Installation Complete!")
    else:
        print("\nAll packages are here. No action needed.")


def check_gpu():
    """
    Checks for GPU availability and prints device details.
    """
    print("\n--- Checking Hardware Accelerator ---")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU Detected: {device_name}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("CUDA Version:", torch.version.cuda)
    else:
        print("No GPU detected! Training will be slow.")
        print("Action: Go to Runtime > Change runtime type > Select T4 GPU")


def setup_environment(pipeline_dir):
    """
    Runs Safe Install
    Checks GPU Status
    """
    print("--- Setting up Colab Environment ---")
      
    # 1. Run the installer
    safe_install(pipeline_dir)
    
    # 2. Check GPU
    check_gpu()        