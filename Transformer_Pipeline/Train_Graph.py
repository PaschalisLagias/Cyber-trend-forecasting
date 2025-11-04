# Import Standard python library imports
import os
import sys
import json
import argparse
import time

# Import Third-party imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import r2_score
import wandb # For experiment logging

# --- Add Project Root to Python Path ---
# This allows us to import from our Transformers/ submodule
# Assumes this script is in 'Transformer_Pipeline/' and root is one level up.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# --- End Path Setup ---

# Local application imports
from Graph_Dataset import CyberThreatGraphDataset
from Models.PDFormer_Wrapper import PDFormerModel
from Transformers.Graph_Transformer.libcity.data.batch import Batch # Import the custom Batch class

def pdformer_collate_fn(batch_list, feature_name):
    """
    Custom collator function required by the PDFormer model.

    This function takes a list of samples from our CyberThreatGraphDataset
    (which are (X, y) tuples of NumPy arrays) and assembles them into
    the special 'Batch' object that the PDFormer model's 'forward' 
    method expects.

    Args:
        batch_list (list[tuple]): A list of (x_sample, y_sample) tuples, 
                                  where each is a NumPy array.
        feature_name (dict): A dictionary defining the keys and data types 
                             for the Batch object, e.g., {'X': 'float', 'y': 'float'}.

    Returns:
        libcity.data.batch.Batch: A single 'Batch' object containing all
                                  samples from the list.
    """
    # Initialise the custom Batch object
    batch = Batch(feature_name)
    
    # Loop through each (x, y) tuple in the list provided by the DataLoader
    for x_sample, y_sample in batch_list:
        # Append the sample to the Batch object
        # The 'append' method expects a tuple/list of features
        batch.append((x_sample, y_sample))
    
    # The 'padding' method (from the original Batch.py) is not needed
    # here because all our time-series windows are of a fixed, uniform length.
    
    # Return the populated Batch object (still on CPU)
    return batch

def main():
    """
    Main entry point for the training and evaluation script.
    """
    
    # --- 1. Configuration Setup ---
    print("--- 1. Setting up Configuration ---")
    parser = argparse.ArgumentParser(description="Main training script for PDFormer model on Cyber Threat data.")
    
    # Add arguments for paths and key hyperparameters
    parser.add_argument('--config_file', type=str, default='pdformer_config.json',
                        help="Path to the JSON config file (relative to this script).")
    parser.add_argument('--data_dir', type=str, default='../Processed_Data/graph',
                        help="Path to the directory with processed data files (.npz, .npy, .pkl).")
    parser.add_argument('--learning_rate', type=float, default=None, 
                        help="Override for learning rate in the config file.")
    parser.add_argument('--epochs', type=int, default=None, 
                        help="Override for number of epochs in the config file.")
    parser.add_argument('--batch_size', type=int, default=None, 
                        help="Override for batch size in the config file.")
    
    args = parser.parse_args()

    # Load the JSON config file
    config_path = os.path.join(SCRIPT_DIR, args.config_file)
    with open(config_path, 'r') as f:
        config = json.load(f)
        print(f"Loaded configuration from {config_path}")

    # Override config with any provided command-line args
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.epochs:
        config['max_epoch'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = str(device) # Add device to config for logging
    print(f"Using device: {device}")

    # --- 2. Weights & Biases Logging Setup ---
    print("\n--- 2. Initialising Weights & Biases ---")
    # Placeholder: Initialise Weights & Biases for experiment tracking
    # wandb.init(
    #     project="Cyber-Threat-Transformer",
    #     name=f"PDFormer_lr{config['learning_rate']}_bs{config['batch_size']}",
    #     config=config
    # )
    print("W&B initialisation outlined.")
    
    # --- 3. Data Setup ---
    print("\n--- 3. Setting up DataLoaders ---")
    
    # Create the Dataset instances
    train_dataset = CyberThreatGraphDataset(data_dir=args.data_dir, split='train')
    val_dataset = CyberThreatGraphDataset(data_dir=args.data_dir, split='val')
    # test_dataset = CyberThreatGraphDataset(data_dir=args.data_dir, split='test')
    
    # Define the feature names and types for the custom Batch object
    # This must match the keys used in the model's forward pass (batch['X'], batch['y'])
    batch_feature_name = {'X': 'float', 'y': 'float'}

    # Create the custom collator function
    # We use a lambda to pass the feature_name dict to the collator
    collator_fn = lambda batch_list: pdformer_collate_fn(batch_list, batch_feature_name)
    
    # Create the DataLoader instances
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True, # Shuffle training data
        num_workers=config.get('num_workers', 0),
        collate_fn=collator_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config['batch_size'],
        shuffle=False, # Do not shuffle validation data
        num_workers=config.get('num_workers', 0),
        collate_fn=collator_fn
    )
    print("DataLoaders created successfully.")

    # --- 4. Model Initialisation ---
    print("\n--- 4. Initialising Model ---")
    
    # Get the dictionary of all static graph artifacts
    # We get this from the training dataset, but it's the same for all splits
    static_features = train_dataset.get_static_features()
    
    # Move all static features (NumPy arrays) to the correct device as Tensors
    # The model's __init__ expects tensors for its internal masks
    # for key, value in static_features.items():
    #     if isinstance(value, np.ndarray):
    #         static_features[key] = torch.from_numpy(value).to(device)
    #     # Note: The scaler object is not a tensor and should remain as-is
    
    # Add necessary features from the dataset to the config for the model
    # config['num_nodes'] = static_features.get('adj_mx').shape[0] # Example
    # config['feature_dim'] = train_dataset.X.shape[-1] # Example
    
    # Initialise our model wrapper
    # model = PDFormerModel(model_config=config, data_feature=static_features).to(device)
    
    # Placeholder: Print model parameter count
    # print(f"Model initialised. Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("Model initialisation outlined.")
    
    # --- 5. Training Components Setup ---
    print("\n--- 5. Setting up Loss and Optimiser ---")
    
    # Define our external Loss Function (MAE / L1 Loss, as agreed)
    # We will calculate this ourselves, ignoring the model's internal .calculate_loss()
    loss_fn = nn.L1Loss() 
    
    # Define the Optimiser (AdamW, as specified in the config)
    # optimizer = optim.AdamW(
    #     model.parameters(), 
    #     lr=config['learning_rate'], 
    #     weight_decay=config['weight_decay']
    # )
    
    # Placeholder: Learning rate scheduler (Cosine, as specified in config)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epoch'])
    
    print(f"Loss function: {loss_fn.__class__.__name__}, Optimiser: AdamW")

    # --- 6. Training & Evaluation Loops ---
    print("\n--- 6. Starting Training & Evaluation ---")
    
    # Placeholder: for epoch in range(config['max_epoch']):
    #     print(f"\n--- Epoch {epoch+1}/{config['max_epoch']} ---")
    #     
    #     # --- Training Loop ---
    #     model.train()
    #     train_epoch_loss = 0.0
    #     
    #     for batch in train_loader: # train_loader yields our custom Batch object
    #         # Move the batch data to the device
    #         batch.to_tensor(device)
    #         
    #         # Get true labels (for our external loss calculation)
    #         y_true = batch['y'] # Shape: (batch_size, output_window, num_nodes, features)
    #
    #         # Forward pass: Get raw predictions
    #         optimizer.zero_grad()
    #         y_predicted = model(batch) # Calls our wrapper's forward method
    #
    #         # Calculate loss (externally)
    #         # We must ensure the shapes match. PDFormer output is (B, T, N, F)
    #         loss = loss_fn(y_predicted, y_true)
    #         
    #         # Backward pass and optimise
    #         loss.backward()
    #         optimizer.step()
    #
    #         train_epoch_loss += loss.item()
    #
    #     # Log training loss for the epoch
    #     avg_train_loss = train_epoch_loss / len(train_loader)
    #     wandb.log({"epoch": epoch, "train_loss_mae": avg_train_loss})
    #     print(f"Epoch {epoch+1} Training Loss (MAE): {avg_train_loss:.6f}")
    #
    #     # --- Validation Loop ---
    #     model.eval()
    #     val_epoch_loss_mae = 0.0
    #     val_epoch_loss_mse = 0.0
    #     all_y_true = []
    #     all_y_pred = []
    #
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             # Move the batch data to the device
    #             batch.to_tensor(device)
    #             y_true = batch['y']
    #             
    #             # Forward pass
    #             y_predicted = model(batch)
    #
    #             # --- Evaluation Metrics (as per Cyber-Threat Paper) ---
    #             # Here we calculate all 4 metrics from the original paper's Table 5
    #             # to ensure our results are comparable.
    #             
    #             # 1. MAE (Mean Absolute Error)
    #             val_epoch_loss_mae += loss_fn(y_predicted, y_true).item()
    #             
    #             # 2. MSE (Mean Squared Error)
    #             mse_fn = nn.MSELoss()
    #             val_epoch_loss_mse += mse_fn(y_predicted, y_true).item()
    #
    #             # For R-squared, we must aggregate all predictions and targets
    #             # to calculate it once at the end of the epoch.
    #             all_y_true.append(y_true.cpu().numpy())
    #             all_y_pred.append(y_predicted.cpu().numpy())
    #             # --- End Metrics ---
    #
    #     # Calculate average epoch losses
    #     avg_val_mae = val_epoch_loss_mae / len(val_loader)
    #     avg_val_mse = val_epoch_loss_mse / len(val_loader)
    #     
    #     # 3. RMSE (Root Mean Squared Error)
    #     avg_val_rmse = np.sqrt(avg_val_mse)
    #
    #     # 4. R-squared (R²)
    #     # Concatenate all batches to calculate R² on the entire validation set
    #     all_y_true = np.concatenate(all_y_true, axis=0)
    #     all_y_pred = np.concatenate(all_y_pred, axis=0)
    #     # Reshape to 2D (samples * time * nodes * feats) for sklearn
    #     all_y_true_flat = all_y_true.reshape(-1, all_y_true.shape[-1])
    #     all_y_pred_flat = all_y_pred.reshape(-1, all_y_pred.shape[-1])
    #     val_r2 = r2_score(all_y_true_flat, all_y_pred_flat)
    #     
    #     # Log all metrics to W&B
    #     wandb.log({
    #         "epoch": epoch,
    #         "val_loss_mae": avg_val_mae,
    #         "val_loss_mse": avg_val_mse,
    #         "val_loss_rmse": avg_val_rmse,
    #         "val_r2_score": val_r2
    #     })
    #     print(f"Epoch {epoch+1} Validation MAE: {avg_val_mae:.6f} | RMSE: {avg_val_rmse:.6f} | R²: {val_r2:.4f}")
    #
    #     # Step the LR scheduler
    #     scheduler.step()
    #
    # print("\n--- Training Complete ---")
    
    print("Training script skeleton outlined.")

if __name__ == '__main__':
    main()# code here