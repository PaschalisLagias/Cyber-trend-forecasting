"""
Cyber_Trend_to_Image_PP.py - Data preprocessing for VisionTS++ pipeline.

This script prepares data for VisionTS++. It can either:
1. Create new preprocessed data in the visionpp directory
2. Symlink to existing vision preprocessed data (if compatible)

VisionTS++ uses the same data format as VisionTS, but processes it differently
(shared image space for multivariate instead of channel-independent).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent


def setup_visionpp_data(use_existing: bool = True, force: bool = False, **preprocessing_args):
    """
    Setup data directory for VisionTS++.

    Args:
        use_existing: If True, copy/link from existing vision data
        force: If True, overwrite existing visionpp data
        **preprocessing_args: Arguments to pass to preprocessing if creating new data
    """
    repo_root = SCRIPT_DIR.parent.parent
    vision_dir = repo_root / "Processed_Data" / "vision"
    visionpp_dir = repo_root / "Processed_Data" / "visionpp"

    print("=" * 60)
    print("VisionTS++ Data Preparation")
    print("=" * 60)

    # Check if visionpp directory already exists
    if visionpp_dir.exists() and not force:
        print(f"\nVisionTS++ data directory already exists: {visionpp_dir}")
        print("Use --force to overwrite, or use existing data.")

        # Check if it has the required files
        required_files = ["train.npz", "val.npz", "test.npz", "metadata.pkl"]
        missing = [f for f in required_files if not (visionpp_dir / f).exists()]
        if missing:
            print(f"WARNING: Missing files: {missing}")
            print("Consider running with --force to recreate data.")
        else:
            print("All required files present. Ready to train!")
        return

    # Create visionpp directory
    visionpp_dir.mkdir(parents=True, exist_ok=True)

    if use_existing and vision_dir.exists():
        print(f"\nCopying data from existing vision preprocessing: {vision_dir}")

        # Check if vision data exists
        required_files = ["train.npz", "val.npz", "test.npz", "metadata.pkl", "feature_names.npy"]
        missing = [f for f in required_files if not (vision_dir / f).exists()]
        if missing:
            print(f"ERROR: Vision data missing files: {missing}")
            print("Run Preprocessing/Cyber_Trend_to_Image.py first.")
            return

        # Copy files
        for f in required_files:
            src = vision_dir / f
            dst = visionpp_dir / f
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  Copied: {f}")

        # Copy optional files
        optional_files = ["scaler.pkl", "pca_model.pkl", "feature_selection.pkl"]
        for f in optional_files:
            src = vision_dir / f
            if src.exists():
                shutil.copy2(src, visionpp_dir / f)
                print(f"  Copied: {f}")

        # Create checkpoints directory
        (visionpp_dir / "checkpoints" / "visiontspp").mkdir(parents=True, exist_ok=True)
        (visionpp_dir / "ckpt").mkdir(parents=True, exist_ok=True)

        print(f"\nData prepared at: {visionpp_dir}")
        print("Ready to train VisionTS++!")

    else:
        print(f"\nCreating new preprocessed data for VisionTS++")

        # Build preprocessing arguments
        preprocess_args = ["--feature-selection", "--n-features", "25"]

        if preprocessing_args.get("no_smoothing"):
            preprocess_args.append("--no-smoothing")
        if preprocessing_args.get("n_features"):
            preprocess_args[-1] = str(preprocessing_args["n_features"])

        # Run preprocessing as subprocess (saves to vision directory)
        preprocess_script = SCRIPT_DIR / "Cyber_Trend_to_Image.py"
        cmd = [sys.executable, str(preprocess_script)] + preprocess_args
        print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
        if result.returncode != 0:
            print(f"ERROR: Preprocessing failed with code {result.returncode}")
            return

        # Then copy to visionpp
        if vision_dir.exists():
            for f in os.listdir(vision_dir):
                src = vision_dir / f
                dst = visionpp_dir / f
                if src.is_file():
                    shutil.copy2(src, dst)
                elif src.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)

        print(f"\nData prepared at: {visionpp_dir}")
        print("Ready to train VisionTS++!")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for VisionTS++ model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--use-existing",
        action="store_true",
        default=True,
        help="Use existing vision preprocessed data if available",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Force creation of new preprocessed data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing visionpp data",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=25,
        help="Number of features to select",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Skip double exponential smoothing",
    )

    args = parser.parse_args()

    setup_visionpp_data(
        use_existing=not args.new,
        force=args.force,
        n_features=args.n_features,
        no_smoothing=args.no_smoothing,
    )


if __name__ == "__main__":
    main()
