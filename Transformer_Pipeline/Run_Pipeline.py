# Import Standard python library imports
import os
import sys
import argparse
import subprocess
import shlex # Used for safely splitting command-line strings

def get_project_root():
    """Finds the project's root directory by looking for the .git folder."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up from Transformer_Pipeline/ to the root
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    if not os.path.exists(os.path.join(project_root, '.git')):
        # Fallback if no git repo found, assume one level up is root
        return project_root
    return project_root

def main():
    """
    Main entry point for the entire experiment pipeline (Preprocessing + Training).

    Script calls the preprocessing and training scripts in the correct order,
    so that data is generated before training begins.
    """

    # --- 1. Define All Paths ---
    PROJECT_ROOT = get_project_root()
    
    # Define processed data directory
    PROCESSED_GRAPH_DIR = os.path.join(PROJECT_ROOT, 'Processed_Data', 'graph')
    
    # Define Default Config
    DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'Transformer_Pipeline', 'pdformer_config.json')

    # Path to Training Script (can run as a file as added sys.path inside it)
    TRAIN_GRAPH_SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'Transformer_Pipeline', 'Train_Graph.py')

    # --- 2. Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Main Experiment Pipeline Runner")

    # Workflow Control
    parser.add_argument('--model', type=str, required=True, choices=['graph', 'vision'],
                        help="Which model pipeline to run.")
    parser.add_argument('--preprocess-only', action='store_true', help="Run only preprocessing.")
    parser.add_argument('--train-only', action='store_true', help="Skip preprocessing check.")
    parser.add_argument('--force-preprocess', action='store_true', help="Force re-run of preprocessing.")

    # Pass-Through Arguments
    parser.add_argument('--pdformer', action='store_true', help="Generate PDFormer artifacts.")
    parser.add_argument('--config_file', type=str, default=DEFAULT_CONFIG_PATH, help="Path to config file.")
    parser.add_argument('--epochs', type=int, default=None, help="Override epochs.")
    parser.add_argument('--learning_rate', type=float, default=None, help="Override learning rate.")
    parser.add_argument('--batch_size', type=int, default=None, help="Override batch size.")

    args = parser.parse_args()

    # --- 3. Execute the Graph Model Pipeline ---
    if args.model == 'graph':
        print(f"--- Starting: Graph Model Pipeline (Root: {PROJECT_ROOT}) ---")

        # --- 3.1 Preprocessing Step ---
        expected_train_file = os.path.join(PROCESSED_GRAPH_DIR, 'train.npz')
        expected_adj_file = os.path.join(PROCESSED_GRAPH_DIR, 'adj_mx.npy')
        
        # Determine if we need to run preprocessing
        data_exists = os.path.exists(expected_train_file) and os.path.exists(expected_adj_file)
        
        run_preprocessing = False
        if args.force_preprocess:
            print("Flag --force-preprocess set. Forcing preprocessing...")
            run_preprocessing = True
        elif not args.train_only and not data_exists:
            print("Processed data not found. Running preprocessing...")
            run_preprocessing = True
        elif args.train_only:
            print("Flag --train-only set. Skipping preprocessing check.")
        else:
            print("Found existing processed data. Skipping preprocessing.")

        if run_preprocessing:
            # UPDATE: Run as a MODULE to support relative imports
            # Use the dot notation: Transformer_Pipeline.Preprocessing.Cyber_Trend_to_Graph
            cmd_preprocess = [
                sys.executable, 
                '-m', 
                'Transformer_Pipeline.Preprocessing.Cyber_Trend_to_Graph'
            ]
            
            if args.pdformer:
                cmd_preprocess.append('--pdformer')

            print(f"Running Module: {' '.join(cmd_preprocess)}")

            try:
                # UPDATE: cwd=PROJECT_ROOT ensures python finds the module package correctly
                subprocess.run(cmd_preprocess, check=True, cwd=PROJECT_ROOT)
                print("Preprocessing complete.")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Preprocessing failed with exit code {e.returncode}.")
                return

        # --- 3.2 Training Step ---
        if not args.preprocess_only:
            print("\n--- Starting: Training Step ---")

            cmd_train = [
                sys.executable,
                TRAIN_GRAPH_SCRIPT_PATH,
                '--config_file', args.config_file,
                '--data_dir', PROCESSED_GRAPH_DIR
            ]

            if args.epochs: cmd_train.extend(['--epochs', str(args.epochs)])
            if args.learning_rate: cmd_train.extend(['--learning_rate', str(args.learning_rate)])
            if args.batch_size: cmd_train.extend(['--batch_size', str(args.batch_size)])

            print(f"Running command: {' '.join(cmd_train)}")

            try:
                # We also run training from PROJECT_ROOT to keep paths consistent
                subprocess.run(cmd_train, check=True, cwd=PROJECT_ROOT)
                print("Training complete.")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Training failed with exit code {e.returncode}.")
                return
        else:
            print("Flag --preprocess-only set. Skipping training.")

        print("\n--- Graph Model Pipeline Finished ---")

    # --- 4. Execute the Vision Model Pipeline ---
    elif args.model == 'vision':
        print("--- Starting: Vision Model Pipeline ---")

        # --- 4.1 Define paths for vision preprocessing and training scripts ---
        PREPROCESS_VISION_SCRIPT = os.path.join(
            PROJECT_ROOT, "Transformer_Pipeline", "Preprocessing", "Cyber_Trend_to_Image.py"
        )
        PROCESSED_VISION_DIR = os.path.join(PROJECT_ROOT, "Processed_Data", "vision")
        TRAIN_VISION_SCRIPT = os.path.join(PROJECT_ROOT, "Transformer_Pipeline", "Train_Vision.py")

        # --- 4.2 Preprocessing Step ---
        expected_vision_file = os.path.join(PROCESSED_VISION_DIR, "train.npy")

        # Check if preprocessing is needed
        data_exists = os.path.exists(expected_vision_file)
        run_preprocessing = False
        if args.force_preprocess:
            print("Flag --force-preprocess set. Forcing preprocessing...")
            run_preprocessing = True
        elif not args.train_only and not data_exists:
            print("Processed vision data not found. Running preprocessing...")
            run_preprocessing = True
        elif args.train_only:
            print("Flag --train-only set. Skipping preprocessing check.")
        else:
            print("Found existing processed vision data. Skipping preprocessing.")

        if run_preprocessing:
            # Build the command for the preprocessing script
            cmd_preprocess = [
                sys.executable,  # Use the current Python interpreter
                PREPROCESS_VISION_SCRIPT,
            ]

            print(f"Running command: {' '.join(cmd_preprocess)}")

            # Call the subprocess
            try:
                subprocess.run(cmd_preprocess, check=True)
                print("Preprocessing complete.")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Vision preprocessing script failed with exit code {e.returncode}.")
                return  # Exit if preprocessing fails

        # --- 4.3 Training Step ---
        if not args.preprocess_only:
            print("\n--- Starting: Vision Training Step ---")

            # Build the command for the training script
            cmd_train = [
                sys.executable,
                TRAIN_VISION_SCRIPT,
            ]
            # Add the optional pass-through arguments if they were provided
            if args.epochs:
                cmd_train.extend(["--epochs", str(args.epochs)])
            if args.learning_rate:
                cmd_train.extend(["--learning_rate", str(args.learning_rate)])
            if args.batch_size:
                cmd_train.extend(["--batch_size", str(args.batch_size)])

            print(f"Running command: {' '.join(cmd_train)}")

            # Call the subprocess
            try:
                subprocess.run(cmd_train, check=True)
                print("Vision training complete.")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Vision training script failed with exit code {e.returncode}.")
                return
        else:
            print("Flag --preprocess-only set. Skipping training.")

        print("\n--- Vision Model Pipeline Finished ---")


if __name__ == '__main__':
    main()
