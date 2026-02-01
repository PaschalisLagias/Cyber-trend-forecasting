# Import Standard python library imports
import os
import sys
import pickle
import argparse

# Import Third-party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Imports from base paper
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Local imports - using project filename conventions
try:
    from .Load_Data import load_cyber_threat_data
except ImportError:
    from Load_Data import load_cyber_threat_data


# Features required for consistency with Graph vignette visualizations
REQUIRED_VIGNETTE_FEATURES = [
    # Validation nodes (Figure 3 equivalents)
    "Password Attack-ALL",
    "Solution_NLP/LLM_Papers",
    "Solution_DATA_BACKUPS_Papers",
    # Continuous trends (Figure 4)
    "Malware-ALL",
    "Vulnerability-ALL",
    "Ransomware-ALL",
    "Papers_Adversarial_Attack",
    # Key gap analysis threats
    "DDoS-ALL",
    "Phishing-ALL",
]


# ------------------- Preprocessing Functions -------------------


def apply_double_exponential_smoothing(df, alpha=0.1, beta=0.1):
    """
    Applies Double Exponential Smoothing (Holt's Linear Trend) to the dataframe.

    This is the same smoothing applied in the Graph pipeline for consistency.

    Args:
        df (pd.DataFrame): Input data (Time x Features).
        alpha (float): Smoothing factor for the level (0 < alpha < 1).
        beta (float): Smoothing factor for the trend (0 < beta < 1).

    Returns:
        pd.DataFrame: Smoothed data with the same shape and index.
    """
    print(f"\n--- Applying Double Exponential Smoothing (alpha={alpha}, beta={beta}) ---")

    df_smoothed = df.copy()
    success_count = 0

    for col in df.columns:
        try:
            model = ExponentialSmoothing(df[col], trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
            df_smoothed[col] = fit.fittedvalues
            success_count += 1
        except Exception as e:
            print(f"Warning: Could not smooth column '{col}': {e}. Keeping raw data.")

    print(f"Smoothed {success_count}/{len(df.columns)} columns.")
    return df_smoothed


def split_data(df_raw, train_split, val_split, min_split_size=None):
    """
    Splits the raw DataFrame chronologically into training, validation, and test sets.

    Args:
        df_raw (pd.DataFrame): The raw DataFrame.
        train_split (float): The proportion of data for the training set (e.g., 0.43).
        val_split (float): The proportion of data for the validation set (e.g., 0.30).
        min_split_size (int, optional): Minimum size for each split. If provided,
            splits will be adjusted to ensure each has at least this many samples.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames for train, val, test sets.
    """
    print("\n--- Step 1: Splitting Data ---")
    n = len(df_raw)

    # Calculate initial split points
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    # Adjust splits if minimum size is specified
    if min_split_size is not None:
        test_size = n - val_end
        val_size = val_end - train_end

        # Adjust test split if too small
        if test_size < min_split_size:
            shortage = min_split_size - test_size
            val_end = val_end - shortage
            print(f"Adjusted val_end by -{shortage} to ensure test has {min_split_size} samples")

        # Adjust val split if too small
        val_size = val_end - train_end
        if val_size < min_split_size:
            shortage = min_split_size - val_size
            train_end = train_end - shortage
            print(f"Adjusted train_end by -{shortage} to ensure val has {min_split_size} samples")

        # Check train split
        if train_end < min_split_size:
            raise ValueError(
                f"Insufficient data: train split has {train_end} samples, "
                f"but need at least {min_split_size}. Total data: {n} samples."
            )

    train_df = df_raw.iloc[:train_end]
    val_df = df_raw.iloc[train_end:val_end]
    test_df = df_raw.iloc[val_end:]

    print(f"Train: {train_df.shape} ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"Val:   {val_df.shape} ({val_df.index.min().date()} to {val_df.index.max().date()})")
    print(f"Test:  {test_df.shape} ({test_df.index.min().date()} to {test_df.index.max().date()})")
    return train_df, val_df, test_df


def split_windowed_data(X, y, train_split, val_split):
    """
    Splits pre-windowed data into training, validation, and test sets.

    This approach windows the full dataset first, then splits the samples.
    This ensures each split has valid samples even with small datasets.

    Args:
        X (np.ndarray): Input windows of shape (num_samples, window_size, num_features).
        y (np.ndarray): Target windows of shape (num_samples, forecast_horizon, num_features).
        train_split (float): The proportion of samples for training (e.g., 0.43).
        val_split (float): The proportion of samples for validation (e.g., 0.30).

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    print("\n--- Splitting Windowed Data ---")
    n = X.shape[0]
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"Train samples: {X_train.shape[0]}")
    print(f"Val samples:   {X_val.shape[0]}")
    print(f"Test samples:  {X_test.shape[0]}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_reference_scaler(train_df, output_dir):
    """
    Computes and saves a MinMaxScaler fitted on training data for reference.

    NOTE: This scaler is NOT applied to the data for VisionTS because VisionTS
    performs its own internal instance normalization. The scaler is saved only
    for analysis, comparison with Graph pipeline, or potential inverse transforms.

    Args:
        train_df (pd.DataFrame): The training data DataFrame.
        output_dir (str): The directory path to save the scaler object.

    Returns:
        MinMaxScaler: The fitted scaler object.
    """
    print("\n--- Step 1: Computing Reference Scaler (NOT applied to data) ---")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df)

    # Save the scaler for reference/analysis
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Reference scaler saved to {scaler_path}")
    print("NOTE: VisionTS handles normalization internally - scaler saved for reference only.")
    return scaler


def select_features_by_pca_importance(train_data, val_data, test_data, n_features=13, output_dir=None):
    """
    Select top N most important original features based on PCA loadings.

    Instead of transforming to PCA space, this identifies which original features
    contribute most to variance and returns those raw features. This preserves
    interpretability while reducing dimensionality.

    Args:
        train_data: Training data array of shape (T_train, n_features)
        val_data: Validation data array of shape (T_val, n_features)
        test_data: Test data array of shape (T_test, n_features)
        n_features: Number of top features to select
        output_dir: Directory to save feature selection info (optional)

    Returns:
        tuple: (train_selected, val_selected, test_selected, selected_indices, feature_importance)
    """
    print(f"\n--- Selecting Top {n_features} Features by PCA Importance ---")

    # Convert DataFrames to numpy if needed
    train_np = train_data.values if hasattr(train_data, 'values') else train_data
    val_np = val_data.values if hasattr(val_data, 'values') else val_data
    test_np = test_data.values if hasattr(test_data, 'values') else test_data

    original_n_features = train_np.shape[1]

    # Fit full PCA to get loadings
    # n_components must be <= min(n_samples, n_features)
    max_components = min(train_np.shape[0], original_n_features, n_features * 2)
    pca = PCA(n_components=max_components)
    pca.fit(train_np)

    # Compute feature importance as max absolute loading across components
    # This identifies features that contribute most to the principal components
    loadings = np.abs(pca.components_)  # Shape: (n_components, n_features)
    feature_importance = loadings.max(axis=0)  # Max loading per feature

    # Select top N features
    selected_indices = np.argsort(feature_importance)[::-1][:n_features]
    selected_indices = np.sort(selected_indices)  # Keep original order

    # Extract selected features (raw values, not transformed)
    train_selected = train_np[:, selected_indices]
    val_selected = val_np[:, selected_indices]
    test_selected = test_np[:, selected_indices]

    print(f"Feature selection: {original_n_features} features → {n_features} selected")
    print(f"Selected feature indices: {selected_indices[:10]}..." if len(selected_indices) > 10 else f"Selected indices: {selected_indices}")
    print(f"Shapes: train={train_selected.shape}, val={val_selected.shape}, test={test_selected.shape}")

    # Save selection info if output directory provided
    if output_dir is not None:
        selection_info = {
            'selected_indices': selected_indices,
            'feature_importance': feature_importance,
            'n_features_selected': n_features,
            'original_n_features': original_n_features,
        }
        selection_path = os.path.join(output_dir, "feature_selection.pkl")
        with open(selection_path, "wb") as f:
            pickle.dump(selection_info, f)
        print(f"Feature selection info saved to {selection_path}")

    return train_selected, val_selected, test_selected, selected_indices, feature_importance


def select_features_hybrid(train_data, val_data, test_data, n_features=224,
                           variance_threshold=0.01, correlation_threshold=0.95,
                           output_dir=None, forced_features=None, all_column_names=None):
    """
    Hybrid feature selection combining variance filtering, correlation removal,
    and PCA importance ranking.

    Steps:
    1. Remove features with variance below threshold (near-constant)
    2. Remove highly correlated features (keep one from each correlated pair)
    3. Select top N by PCA importance from remaining features
    4. If forced_features provided, ensure they are included (replacing lowest-importance features)

    Args:
        train_data: Training data array of shape (T_train, n_features)
        val_data: Validation data array of shape (T_val, n_features)
        test_data: Test data array of shape (T_test, n_features)
        n_features: Target number of features to select (default: 224, VisionTS++ max)
        variance_threshold: Minimum variance for feature inclusion (default: 0.01)
        correlation_threshold: Maximum correlation before removing redundant feature (default: 0.95)
        output_dir: Directory to save feature selection info (optional)
        forced_features: List of feature names that must be included (optional)
        all_column_names: Array of all column names to map forced feature names to indices (required if forced_features is provided)

    Returns:
        tuple: (train_selected, val_selected, test_selected, selected_indices, selection_info)
    """
    print(f"\n--- Hybrid Feature Selection (target: {n_features} features) ---")

    # Convert to numpy
    train_np = train_data.values if hasattr(train_data, 'values') else train_data
    val_np = val_data.values if hasattr(val_data, 'values') else val_data
    test_np = test_data.values if hasattr(test_data, 'values') else test_data
    original_n = train_np.shape[1]

    # Step 0: Identify forced feature indices (if any)
    forced_indices = []
    forced_names_found = []
    if forced_features is not None and all_column_names is not None:
        all_names_list = list(all_column_names)
        for feat in forced_features:
            if feat in all_names_list:
                forced_indices.append(all_names_list.index(feat))
                forced_names_found.append(feat)
            else:
                print(f"  Warning: Forced feature '{feat}' not found in data")
        print(f"  Step 0 - Forced features: {len(forced_names_found)} of {len(forced_features)} found")
        if forced_names_found:
            print(f"    Found: {forced_names_found}")

    # Step 1: Variance filtering
    variances = np.var(train_np, axis=0)
    variance_mask = variances >= variance_threshold
    n_after_var = variance_mask.sum()
    print(f"  Step 1 - Variance filter (threshold={variance_threshold}): {original_n} -> {n_after_var}")

    # Step 2: Correlation filtering (on variance-filtered features)
    train_filtered = train_np[:, variance_mask]
    corr_matrix = np.corrcoef(train_filtered.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)  # Handle NaN

    # Find highly correlated pairs and keep first occurrence
    to_keep = np.ones(n_after_var, dtype=bool)
    for i in range(n_after_var):
        if not to_keep[i]:
            continue
        for j in range(i + 1, n_after_var):
            if abs(corr_matrix[i, j]) > correlation_threshold:
                to_keep[j] = False

    n_after_corr = to_keep.sum()
    print(f"  Step 2 - Correlation filter (threshold={correlation_threshold}): {n_after_var} -> {n_after_corr}")

    # Map back to original indices
    variance_indices = np.where(variance_mask)[0]
    filtered_indices = variance_indices[to_keep]

    # Step 3: PCA importance on filtered features
    train_for_pca = train_np[:, filtered_indices]

    if n_after_corr <= n_features and not forced_indices:
        # Use all remaining features
        selected_indices = filtered_indices
        print(f"  Step 3 - Using all {n_after_corr} remaining features (< target {n_features})")
    else:
        # Apply PCA importance ranking
        max_components = min(train_for_pca.shape[0], n_after_corr, n_features * 2)
        pca = PCA(n_components=max_components)
        pca.fit(train_for_pca)

        loadings = np.abs(pca.components_)
        feature_importance = loadings.max(axis=0)

        if forced_indices:
            # Step 4: Force-include required features
            forced_not_filtered = [idx for idx in forced_indices if idx not in filtered_indices]

            if forced_not_filtered:
                print(f"  Note: {len(forced_not_filtered)} forced features didn't pass filters - including anyway")

            # All forced indices (whether filtered or not)
            all_forced = set(forced_indices)

            # Get remaining slots from filtered features (excluding forced ones)
            remaining_slots = n_features - len(forced_indices)

            if remaining_slots > 0:
                # Rank filtered features by importance, excluding forced ones
                filtered_idx_to_importance = {}
                for i, orig_idx in enumerate(filtered_indices):
                    if orig_idx not in all_forced:
                        filtered_idx_to_importance[orig_idx] = feature_importance[i]

                # Sort by importance and take top remaining_slots
                sorted_by_importance = sorted(filtered_idx_to_importance.items(),
                                              key=lambda x: x[1], reverse=True)
                top_remaining = [idx for idx, _ in sorted_by_importance[:remaining_slots]]

                # Combine forced + top remaining, then sort to maintain original order
                selected_indices = np.array(sorted(list(all_forced) + top_remaining))
            else:
                # More forced features than n_features - just use forced
                selected_indices = np.array(sorted(forced_indices))

            print(f"  Step 3 - Forced inclusion: {len(forced_indices)} forced + {remaining_slots} by importance = {len(selected_indices)}")
        else:
            top_k = np.argsort(feature_importance)[::-1][:n_features]
            selected_indices = filtered_indices[np.sort(top_k)]
            print(f"  Step 3 - PCA importance: {n_after_corr} -> {n_features}")

    # Extract selected features
    train_selected = train_np[:, selected_indices]
    val_selected = val_np[:, selected_indices]
    test_selected = test_np[:, selected_indices]

    print(f"  Final: {original_n} -> {len(selected_indices)} features")

    # Build selection info
    selection_info = {
        'selected_indices': selected_indices,
        'n_features_selected': len(selected_indices),
        'original_n_features': original_n,
        'variance_threshold': variance_threshold,
        'correlation_threshold': correlation_threshold,
        'n_after_variance_filter': int(n_after_var),
        'n_after_correlation_filter': int(n_after_corr),
        'method': 'hybrid',
    }

    # Save selection info if output directory provided
    if output_dir is not None:
        selection_path = os.path.join(output_dir, "feature_selection.pkl")
        with open(selection_path, "wb") as f:
            pickle.dump(selection_info, f)
        print(f"Feature selection info saved to {selection_path}")

    return train_selected, val_selected, test_selected, selected_indices, selection_info


def fit_and_apply_pca(train_data, val_data, test_data, variance_ratio=0.95, output_dir=None):
    """
    Fit PCA on training data and transform all splits.

    PCA is fit ONLY on training data to prevent data leakage. The same
    transformation is then applied to validation and test sets.

    Args:
        train_data: Training data array of shape (T_train, n_features)
        val_data: Validation data array of shape (T_val, n_features)
        test_data: Test data array of shape (T_test, n_features)
        variance_ratio: Target cumulative explained variance (e.g., 0.95 for 95%)
        output_dir: Directory to save PCA model (optional)

    Returns:
        tuple: (train_reduced, val_reduced, test_reduced, pca_model)
    """
    print(f"\n--- Applying PCA (target variance: {variance_ratio:.0%}) ---")

    # Convert DataFrames to numpy if needed
    if hasattr(train_data, 'values'):
        train_data = train_data.values
    if hasattr(val_data, 'values'):
        val_data = val_data.values
    if hasattr(test_data, 'values'):
        test_data = test_data.values

    original_features = train_data.shape[1]

    # Fit PCA on training data only
    pca = PCA(n_components=variance_ratio, svd_solver='full')
    train_reduced = pca.fit_transform(train_data)

    # Transform validation and test with the same PCA
    val_reduced = pca.transform(val_data)
    test_reduced = pca.transform(test_data)

    n_components = pca.n_components_
    explained_var = pca.explained_variance_ratio_.sum()

    print(f"PCA reduction: {original_features} features → {n_components} components")
    print(f"Explained variance: {explained_var:.4f} ({explained_var:.1%})")
    print(f"Shapes: train={train_reduced.shape}, val={val_reduced.shape}, test={test_reduced.shape}")

    # Save PCA model if output directory provided
    if output_dir is not None:
        pca_path = os.path.join(output_dir, "pca_model.pkl")
        with open(pca_path, "wb") as f:
            pickle.dump(pca, f)
        print(f"PCA model saved to {pca_path}")

    return train_reduced, val_reduced, test_reduced, pca


def create_sliding_windows(data, window_size, forecast_horizon):
    """
    Creates sliding windows from data for model input (X) and target (y).

    Unlike the Graph pipeline which outputs 4D tensors, this outputs 3D tensors
    as required by VisionTS: (samples, time_steps, features).

    Args:
        data (np.ndarray or pd.DataFrame): The time-series data (time x features).
        window_size (int): The number of past time steps to use as input.
        forecast_horizon (int): The number of future time steps to predict.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (Input): Shape (num_samples, window_size, num_features)
            - y (Target): Shape (num_samples, forecast_horizon, num_features)
    """
    print("\n--- Step 2: Creating Sliding Windows ---")

    if isinstance(data, pd.DataFrame):
        data = data.values

    num_samples = data.shape[0] - window_size - forecast_horizon + 1
    num_features = data.shape[1]

    if num_samples <= 0:
        raise ValueError(
            f"Insufficient data for windowing. Data length: {data.shape[0]}, "
            f"Window: {window_size}, Horizon: {forecast_horizon}. "
            f"Need at least {window_size + forecast_horizon} time steps."
        )

    # VisionTS expects 3D tensors: (samples, time, features)
    X = np.zeros((num_samples, window_size, num_features), dtype=np.float32)
    y = np.zeros((num_samples, forecast_horizon, num_features), dtype=np.float32)

    for i in range(num_samples):
        X[i] = data[i : i + window_size]
        y[i] = data[i + window_size : i + window_size + forecast_horizon]

    print(f"Generated shapes: X={X.shape}, y={y.shape}")
    print(f"  X: (samples={num_samples}, context_len={window_size}, nvars={num_features})")
    print(f"  y: (samples={num_samples}, pred_len={forecast_horizon}, nvars={num_features})")
    return X, y


def save_processed_data(output_dir, X_train, y_train, X_val, y_val, X_test, y_test, feature_names, metadata=None):
    """
    Saves all processed, windowed data into files.

    Args:
        output_dir (str): The directory path to save the processed files.
        X_train, y_train: Training input and target sequences.
        X_val, y_val: Validation input and target sequences.
        X_test, y_test: Test input and target sequences.
        feature_names (np.ndarray): Array of original column/feature names.
        metadata (dict, optional): Additional metadata to save.

    Returns:
        None
    """
    print("\n--- Step 4: Saving Processed Data ---")

    # Save windowed data as .npz files (same format as Graph pipeline)
    train_path = os.path.join(output_dir, "train.npz")
    val_path = os.path.join(output_dir, "val.npz")
    test_path = os.path.join(output_dir, "test.npz")

    np.savez_compressed(train_path, x=X_train, y=y_train)
    np.savez_compressed(val_path, x=X_val, y=y_val)
    np.savez_compressed(test_path, x=X_test, y=y_test)

    print(f"Saved train.npz: X={X_train.shape}, y={y_train.shape}")
    print(f"Saved val.npz:   X={X_val.shape}, y={y_val.shape}")
    print(f"Saved test.npz:  X={X_test.shape}, y={y_test.shape}")

    # Save feature names
    feature_names_path = os.path.join(output_dir, "feature_names.npy")
    np.save(feature_names_path, feature_names)
    print(f"Saved feature_names.npy: {len(feature_names)} features")

    # Save metadata if provided
    if metadata:
        metadata_path = os.path.join(output_dir, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        print(f"Saved metadata.pkl")

    print("Saving complete.")


# ------------------- Main Function -------------------


def main():
    """
    Main function for the Vision preprocessing pipeline.

    This preprocesses cyber threat data for the VisionTS model:
    - Applies Double Exponential Smoothing (same as Graph pipeline)
    - Optionally applies PCA for dimensionality reduction (default: enabled)
    - Does NOT apply external normalization (VisionTS handles this internally)
    - Creates 3D windowed tensors (samples, time, features)

    Pipeline flow:
    1. Load raw data
    2. Apply smoothing (optional)
    3. Split 2D data chronologically (train/val/test)
    4. Apply PCA - fit on train only, transform all (prevents data leakage)
    5. Create sliding windows for each split
    6. Save processed data
    """
    # --- Configuration - Argument Parsing ---
    parser = argparse.ArgumentParser(description="Data preprocessing script for VisionTS-based time-series models.")
    parser.add_argument("--no-smoothing", action="store_true", help="Skip Double Exponential Smoothing.")
    parser.add_argument("--no-pca", action="store_true", help="Skip PCA dimensionality reduction.")
    parser.add_argument("--pca-variance", type=float, default=0.95, help="PCA variance ratio to retain (default: 0.95)")
    parser.add_argument("--feature-selection", action="store_true", help="Use PCA-based feature selection instead of PCA transformation. Selects top N most important original features.")
    parser.add_argument("--hybrid-selection", action="store_true", help="Use hybrid feature selection (variance + correlation + PCA importance)")
    parser.add_argument("--n-features", type=int, default=224, help="Number of features to select when using --feature-selection or --hybrid-selection (default: 224)")
    parser.add_argument("--variance-threshold", type=float, default=0.01, help="Minimum variance for feature inclusion in hybrid selection (default: 0.01)")
    parser.add_argument("--correlation-threshold", type=float, default=0.95, help="Maximum correlation before removing redundant feature in hybrid selection (default: 0.95)")
    parser.add_argument("--alpha", type=float, default=0.1, help="DES smoothing level parameter (default: 0.1)")
    parser.add_argument("--beta", type=float, default=0.1, help="DES smoothing trend parameter (default: 0.1)")
    parser.add_argument("--window-first", action="store_true", help="Window entire dataset first, then split (more training samples)")
    parser.add_argument("--context-len", type=int, default=36, help="Context window size in months (default: 36)")
    parser.add_argument("--pred-len", type=int, default=36, help="Prediction horizon in months (default: 36)")
    parser.add_argument("--force-vignette-features", action="store_true",
                        help="Force inclusion of features required for vignette comparison with Graph pipeline")
    args = parser.parse_args()

    # --- Configuration - CONSTANTS (Aligned with B-MTGNN Paper) ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Path handling for sibling directories
    RAW_DATA_FILE = os.path.join(SCRIPT_DIR, "../../Data_Preparation/Cyber_Trend_Forecasting_All.csv")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../../Processed_Data/vision")

    # --- Experimental Settings [Source: Paper Section 3.5] ---
    TRAIN_SPLIT = 0.43
    VAL_SPLIT = 0.30
    # Test Split is implicitly 0.27 (remainder)

    # Model input/output settings
    WINDOW_SIZE = args.context_len  # Months History (context_len)
    FORECAST_HORIZON = args.pred_len  # Months Forecast (pred_len)

    # Dimensionality reduction settings
    APPLY_HYBRID_SELECTION = args.hybrid_selection
    APPLY_FEATURE_SELECTION = args.feature_selection and not args.hybrid_selection
    APPLY_PCA = not args.no_pca and not args.feature_selection and not args.hybrid_selection
    PCA_VARIANCE = args.pca_variance
    N_FEATURES = args.n_features
    VARIANCE_THRESHOLD = args.variance_threshold
    CORRELATION_THRESHOLD = args.correlation_threshold

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print("=" * 60)
    print("Vision Pipeline Preprocessing")
    print("=" * 60)
    print(f"\n--- Configuration ---")
    print(f"Context Window (input):  {WINDOW_SIZE} months")
    print(f"Forecast Horizon (pred): {FORECAST_HORIZON} months")
    print(f"Splits: Train={TRAIN_SPLIT:.0%}, Val={VAL_SPLIT:.0%}, Test={1 - (TRAIN_SPLIT + VAL_SPLIT):.0%}")
    print(f"Windowing Strategy: {'Window-First (more samples)' if args.window_first else 'Split-First (standard)'}")
    if APPLY_HYBRID_SELECTION:
        print(f"Dimensionality Reduction: Hybrid Selection (target {N_FEATURES} features)")
        print(f"  - Variance threshold: {VARIANCE_THRESHOLD}")
        print(f"  - Correlation threshold: {CORRELATION_THRESHOLD}")
    elif APPLY_FEATURE_SELECTION:
        print(f"Dimensionality Reduction: Feature Selection (top {N_FEATURES} features by PCA importance)")
    elif APPLY_PCA:
        print(f"Dimensionality Reduction: PCA (variance={PCA_VARIANCE:.0%})")
    else:
        print(f"Dimensionality Reduction: Disabled")
    print(f"Output Directory: {OUTPUT_DIR}")

    # --- Step 0: Load Data ---
    print("\n--- Step 0: Loading Raw Data ---")
    df_raw = load_cyber_threat_data(RAW_DATA_FILE)
    if df_raw is None:
        print("Aborting: Could not load data.")
        return

    original_features = df_raw.shape[1]

    # --- Step 0.5: Apply Double Exponential Smoothing ---
    if args.no_smoothing:
        print("\n--- Skipping Double Exponential Smoothing (--no-smoothing flag) ---")
        df_processed = df_raw
    else:
        df_processed = apply_double_exponential_smoothing(df_raw, alpha=args.alpha, beta=args.beta)

    # Initialize variables used in both branches
    pca_model = None
    pca_n_components = None
    pca_explained_variance = None
    selected_indices = None
    selected_feature_names = None

    if args.window_first:
        # =====================================================================
        # WINDOW-FIRST APPROACH: Window entire dataset, then split samples
        # This maximizes training data by creating overlapping windows across
        # the full dataset, then splitting windows chronologically.
        # =====================================================================
        print("\n" + "=" * 60)
        print("Using WINDOW-FIRST approach (maximum training samples)")
        print("=" * 60)

        # --- Step 1: Apply Dimensionality Reduction on ALL data ---
        # For feature selection: fit PCA on train portion (first 43%) to avoid leakage
        train_portion_end = int(len(df_processed) * TRAIN_SPLIT)

        if APPLY_HYBRID_SELECTION:
            print(f"\n--- Hybrid Feature Selection (fit on first {TRAIN_SPLIT:.0%} of data) ---")
            # Use train portion for fitting, then apply to all data
            train_portion = df_processed.iloc[:train_portion_end]
            val_portion = df_processed.iloc[train_portion_end:int(len(df_processed) * (TRAIN_SPLIT + VAL_SPLIT))]
            test_portion = df_processed.iloc[int(len(df_processed) * (TRAIN_SPLIT + VAL_SPLIT)):]

            # Get forced features if requested
            forced_feats = REQUIRED_VIGNETTE_FEATURES if args.force_vignette_features else None

            # Get selected features using hybrid method
            _, _, _, selected_indices, selection_info = select_features_hybrid(
                train_portion, val_portion, test_portion,
                n_features=N_FEATURES,
                variance_threshold=VARIANCE_THRESHOLD,
                correlation_threshold=CORRELATION_THRESHOLD,
                output_dir=OUTPUT_DIR,
                forced_features=forced_feats,
                all_column_names=df_raw.columns.values
            )

            # Apply selection to ALL data
            all_data = df_processed.values[:, selected_indices]
            num_features = len(selected_indices)
            selected_feature_names = df_raw.columns.values[selected_indices]
            print(f"Selected features: {list(selected_feature_names[:5])}..." if len(selected_feature_names) > 5 else f"Selected features: {list(selected_feature_names)}")

        elif APPLY_FEATURE_SELECTION:
            print(f"\n--- Feature Selection (fit on first {TRAIN_SPLIT:.0%} of data) ---")
            # Fit PCA on train portion only
            train_portion = df_processed.iloc[:train_portion_end]
            train_np = train_portion.values if hasattr(train_portion, 'values') else train_portion

            # Compute feature importance from train portion
            max_components = min(train_np.shape[0], original_features, N_FEATURES * 2)
            pca = PCA(n_components=max_components)
            pca.fit(train_np)

            loadings = np.abs(pca.components_)
            feature_importance = loadings.max(axis=0)
            selected_indices = np.argsort(feature_importance)[::-1][:N_FEATURES]
            selected_indices = np.sort(selected_indices)

            # Apply selection to ALL data
            all_data = df_processed.values[:, selected_indices]
            num_features = N_FEATURES
            selected_feature_names = df_raw.columns.values[selected_indices]
            print(f"Selected {N_FEATURES} features from {original_features}")
            print(f"Selected features: {list(selected_feature_names[:5])}..." if len(selected_feature_names) > 5 else f"Selected features: {list(selected_feature_names)}")

            # Save feature selection info
            selection_info = {
                'selected_indices': selected_indices,
                'feature_importance': feature_importance,
                'n_features_selected': N_FEATURES,
                'original_n_features': original_features,
            }
            selection_path = os.path.join(OUTPUT_DIR, "feature_selection.pkl")
            with open(selection_path, "wb") as f:
                pickle.dump(selection_info, f)

        elif APPLY_PCA:
            print(f"\n--- PCA (fit on first {TRAIN_SPLIT:.0%} of data) ---")
            train_portion = df_processed.iloc[:train_portion_end].values
            pca_model = PCA(n_components=PCA_VARIANCE, svd_solver='full')
            pca_model.fit(train_portion)

            all_data = pca_model.transform(df_processed.values)
            pca_n_components = pca_model.n_components_
            pca_explained_variance = float(pca_model.explained_variance_ratio_.sum())
            num_features = pca_n_components
            print(f"PCA: {original_features} → {pca_n_components} components ({pca_explained_variance:.1%} variance)")

            # Save PCA model
            pca_path = os.path.join(OUTPUT_DIR, "pca_model.pkl")
            with open(pca_path, "wb") as f:
                pickle.dump(pca_model, f)
        else:
            print("\n--- Skipping dimensionality reduction ---")
            all_data = df_processed.values
            num_features = original_features

        # --- Step 2: Create sliding windows on ALL data ---
        print("\n--- Creating Sliding Windows on Full Dataset ---")
        X_all, y_all = create_sliding_windows(all_data, WINDOW_SIZE, FORECAST_HORIZON)
        print(f"Total windows created: {X_all.shape[0]}")

        # --- Step 3: Split windowed data chronologically ---
        X_train, y_train, X_val, y_val, X_test, y_test = split_windowed_data(
            X_all, y_all, TRAIN_SPLIT, VAL_SPLIT
        )

        # Compute reference scaler from train portion of original data
        train_df = df_processed.iloc[:train_portion_end]
        _ = compute_reference_scaler(train_df, OUTPUT_DIR)

    else:
        # =====================================================================
        # SPLIT-FIRST APPROACH (Standard): Split data, then window each split
        # This is the traditional approach that ensures strict temporal separation.
        # =====================================================================

        # --- Step 1: Split 2D data chronologically (BEFORE windowing) ---
        # This ensures PCA is fit only on training data (no data leakage)
        # Minimum split size = window + horizon to ensure at least 1 sample per split
        min_split_size = WINDOW_SIZE + FORECAST_HORIZON
        train_df, val_df, test_df = split_data(df_processed, TRAIN_SPLIT, VAL_SPLIT, min_split_size=min_split_size)

        # --- Step 2: Compute Reference Scaler (fitted on training data, NOT applied) ---
        _ = compute_reference_scaler(train_df, OUTPUT_DIR)

        # --- Step 3: Apply Dimensionality Reduction (if enabled) ---
        if APPLY_HYBRID_SELECTION:
            # Get forced features if requested
            forced_feats = REQUIRED_VIGNETTE_FEATURES if args.force_vignette_features else None

            # Hybrid feature selection: variance + correlation + PCA importance
            train_data, val_data, test_data, selected_indices, _ = select_features_hybrid(
                train_df, val_df, test_df,
                n_features=N_FEATURES,
                variance_threshold=VARIANCE_THRESHOLD,
                correlation_threshold=CORRELATION_THRESHOLD,
                output_dir=OUTPUT_DIR,
                forced_features=forced_feats,
                all_column_names=df_raw.columns.values
            )
            num_features = len(selected_indices)
            selected_feature_names = df_raw.columns.values[selected_indices]
            print(f"Selected features: {list(selected_feature_names[:5])}..." if len(selected_feature_names) > 5 else f"Selected features: {list(selected_feature_names)}")
        elif APPLY_FEATURE_SELECTION:
            # Feature selection: select top N original features by PCA importance
            train_data, val_data, test_data, selected_indices, _ = select_features_by_pca_importance(
                train_df, val_df, test_df,
                n_features=N_FEATURES,
                output_dir=OUTPUT_DIR
            )
            num_features = N_FEATURES
            selected_feature_names = df_raw.columns.values[selected_indices]
            print(f"Selected features: {list(selected_feature_names[:5])}..." if len(selected_feature_names) > 5 else f"Selected features: {list(selected_feature_names)}")
        elif APPLY_PCA:
            # PCA transformation
            train_data, val_data, test_data, pca_model = fit_and_apply_pca(
                train_df, val_df, test_df,
                variance_ratio=PCA_VARIANCE,
                output_dir=OUTPUT_DIR
            )
            pca_n_components = pca_model.n_components_
            pca_explained_variance = float(pca_model.explained_variance_ratio_.sum())
            num_features = pca_n_components
        else:
            print("\n--- Skipping dimensionality reduction (--no-pca flag) ---")
            train_data = train_df.values if hasattr(train_df, 'values') else train_df
            val_data = val_df.values if hasattr(val_df, 'values') else val_df
            test_data = test_df.values if hasattr(test_df, 'values') else test_df
            num_features = original_features

        # --- Step 4: Create Sliding Windows for each split ---
        print("\n--- Creating Sliding Windows for Train Split ---")
        X_train, y_train = create_sliding_windows(train_data, WINDOW_SIZE, FORECAST_HORIZON)

        print("\n--- Creating Sliding Windows for Val Split ---")
        X_val, y_val = create_sliding_windows(val_data, WINDOW_SIZE, FORECAST_HORIZON)

        print("\n--- Creating Sliding Windows for Test Split ---")
        X_test, y_test = create_sliding_windows(test_data, WINDOW_SIZE, FORECAST_HORIZON)

    total_samples = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    print(f"\nTotal samples: {total_samples} (train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]})")

    # --- Step 5: Save Processed Data ---
    metadata = {
        "window_size": WINDOW_SIZE,
        "forecast_horizon": FORECAST_HORIZON,
        "train_split": TRAIN_SPLIT,
        "val_split": VAL_SPLIT,
        "original_num_features": original_features,
        "num_features": num_features,
        "total_samples": total_samples,
        "window_first": args.window_first,
        "smoothing_applied": not args.no_smoothing,
        "smoothing_alpha": args.alpha if not args.no_smoothing else None,
        "smoothing_beta": args.beta if not args.no_smoothing else None,
        "pca_applied": APPLY_PCA,
        "pca_n_components": pca_n_components,
        "pca_variance_target": PCA_VARIANCE if APPLY_PCA else None,
        "pca_explained_variance": pca_explained_variance,
        "feature_selection_applied": APPLY_FEATURE_SELECTION,
        "hybrid_selection_applied": APPLY_HYBRID_SELECTION,
        "variance_threshold": VARIANCE_THRESHOLD if APPLY_HYBRID_SELECTION else None,
        "correlation_threshold": CORRELATION_THRESHOLD if APPLY_HYBRID_SELECTION else None,
        "selected_feature_indices": selected_indices.tolist() if selected_indices is not None else None,
        "selected_feature_names": list(selected_feature_names) if selected_feature_names is not None else None,
        "date_range": {
            "full": (str(df_processed.index.min().date()), str(df_processed.index.max().date())),
        },
    }

    # Use selected feature names if feature selection was applied, otherwise original names
    feature_names_to_save = selected_feature_names if selected_feature_names is not None else df_raw.columns.values

    save_processed_data(
        OUTPUT_DIR,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_names=feature_names_to_save,
        metadata=metadata,
    )

    print("\n" + "=" * 60)
    print("Vision Pipeline Preprocessing Complete")
    print("=" * 60)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("Files created:")
    print("  - train.npz, val.npz, test.npz (windowed data)")
    print("  - scaler.pkl (reference scaler - NOT applied to data)")
    print("  - feature_names.npy (selected/original column names)")
    print("  - metadata.pkl (preprocessing parameters)")
    if APPLY_HYBRID_SELECTION:
        print("  - feature_selection.pkl (hybrid feature selection info)")
        print(f"\nHybrid Selection Summary: {original_features} -> {num_features} features")
        print(f"  Variance threshold: {VARIANCE_THRESHOLD}")
        print(f"  Correlation threshold: {CORRELATION_THRESHOLD}")
    elif APPLY_FEATURE_SELECTION:
        print("  - feature_selection.pkl (feature selection info)")
        print(f"\nFeature Selection Summary: {original_features} -> {num_features} features (raw values preserved)")
    elif APPLY_PCA:
        print("  - pca_model.pkl (fitted PCA transformer)")
        print(f"\nPCA Summary: {original_features} -> {pca_n_components} features ({pca_explained_variance:.1%} variance)")


if __name__ == "__main__":
    main()
