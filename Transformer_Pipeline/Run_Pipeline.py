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
    # A simple check (optional, but good)
    if not os.path.exists(os.path.join(project_root, '.git')):
        print("Warning: Could not automatically find project root. Assuming '.' is the root.")
        return "."
    return project_root

def main():
    """
    Main entry point for the entire experiment pipeline (Preprocessing + Training).

    Script calls the preprocessing and training scripts in the correct order,
    so that data is generated before training begins.
    """

    # --- 1. Define All Paths ---
    # Get the project root so we can build absolute paths
    # This makes the script runnable from any directory
    PROJECT_ROOT = get_project_root()

    # Path to the preprocessing script
    PREPROCESS_SCRIPT_PATH = os.path.join(
        PROJECT_ROOT,
        'Transformer_Pipeline',
        'Preprocessing',
        'Cyber_Trend_to_Graph.py'
    )
    # Path to the graph model training script
    TRAIN_GRAPH_SCRIPT_PATH = os.path.join(
        PROJECT_ROOT,
        'Transformer_Pipeline',
        'Train_Graph.py'
    )
    # Path to the processed graph data
    PROCESSED_GRAPH_DIR = os.path.join(
        PROJECT_ROOT,
        'Processed_Data',
        'graph'
    )

    # Path to the default config file
    DEFAULT_CONFIG_PATH = os.path.join(
        PROJECT_ROOT,
        'Transformer_Pipeline',
        'pdformer_config.json'
    )

    # --- 2. Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Main Experiment Pipeline Runner")

    # --- Workflow Control Arguments ---
    parser.add_argument('--model', type=str, required=True, choices=['graph', 'vision'],
                        help="Which model pipeline to run: 'graph' or 'vision'.")
    parser.add_argument('--preprocess-only', action='store_true',
                        help="Run *only* the preprocessing step, then exit.")
    parser.add_argument('--train-only', action='store_true',
                        help="*Skip* the preprocessing check and run training immediately. "
                             "Assumes data files already exist.")
    parser.add_argument('--force-preprocess', action='store_true',
                        help="Force the preprocessing script to re-run, even if "
                             "processed data files already exist.")

    # --- Pass-Through Arguments for Preprocessing ---
    parser.add_argument('--pdformer', action='store_true',
                        help="[Pass-through] Generate the extra PDFormer-specific artifacts "
                             "(DTW, shortest paths, etc.) during preprocessing.")

    # --- Pass-Through Arguments for Training ---
    parser.add_argument('--config_file', type=str, default=DEFAULT_CONFIG_PATH,
                        help=f"[Pass-through] Path to the .json config file. "
                             f"Defaults to: {DEFAULT_CONFIG_PATH}")
    parser.add_argument('--epochs', type=int, default=None,
                        help="[Pass-through] Override the number of training epochs in the config.")
    parser.add_argument('--learning_rate', type=float, default=None,
                        help="[Pass-through] Override the learning rate in the config.")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="[Pass-through] Override the batch size in the config.")

    args = parser.parse_args()

    # --- 3. Execute the Graph Model Pipeline ---
    if args.model == 'graph':
        print("--- Starting: Graph Model Pipeline ---")

        # --- 3.1 Preprocessing Step ---

        # Define the key files we expect preprocessing to create
        expected_train_file = os.path.join(PROCESSED_GRAPH_DIR, 'train.npz')
        expected_adj_file = os.path.join(PROCESSED_GRAPH_DIR, 'adj_mx.npy')

        # Check if preprocessing is needed
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
            # Build the command for the preprocessing script
            cmd_preprocess = [
                sys.executable, # Use the current Python interpreter
                PREPROCESS_SCRIPT_PATH
            ]
            # Add the --pdformer flag *only if* it was passed to this script
            if args.pdformer:
                cmd_preprocess.append('--pdformer')

            print(f"Running command: {' '.join(cmd_preprocess)}")

            # Call the subprocess
            try:
                subprocess.run(cmd_preprocess, check=True)
                print("Preprocessing complete.")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Preprocessing script failed with exit code {e.returncode}.")
                return # Exit if preprocessing fails

        # --- 3.2 Training Step ---

        if not args.preprocess_only:
            print("\n--- Starting: Training Step ---")

            # Build the command for the training script
            cmd_train = [
                sys.executable,
                TRAIN_GRAPH_SCRIPT_PATH,
                '--config_file', args.config_file,
                '--data_dir', PROCESSED_GRAPH_DIR
            ]

            # Add the optional pass-through arguments if they were provided
            # This is how we override the JSON config from this main script
            if args.epochs:
                cmd_train.extend(['--epochs', str(args.epochs)])
            if args.learning_rate:
                cmd_train.extend(['--learning_rate', str(args.learning_rate)])
            if args.batch_size:
                cmd_train.extend(['--batch_size', str(args.batch_size)])

            print(f"Running command: {' '.join(cmd_train)}")

            # Call the subprocess
            try:
                subprocess.run(cmd_train, check=True)
                print("Training complete.")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Training script failed with exit code {e.returncode}.")
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
