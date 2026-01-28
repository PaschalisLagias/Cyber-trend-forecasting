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
import wandb 

# --- Add Project Root to Python Path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Local application imports
from Cyber_Trend_Graph_Config import PDFormerConfig
from Cyber_Trend_Graph_Dataset import CyberThreatGraphDataset
from Models.PDFormer_Wrapper import PDFormer_Wrapper
from Transformers.Graph_Transformer.libcity.data.batch import Batch 

def pdformer_collate_fn(batch_list, feature_name):
    """
    Custom collator: Converts list of NumPy samples into a libcity Batch object.
    """
    batch = Batch(feature_name)
    for x_sample, y_sample in batch_list:
        batch.append((x_sample, y_sample))
    return batch

def compute_metrics(all_preds, all_targets):
    """
    Computes RSE and RAE as defined in Eq. 10 & 11 of original paper.
    
    Args:
        all_preds (torch.Tensor): Shape (N_samples, Time, Nodes, 1)
        all_targets (torch.Tensor): Shape (N_samples, Time, Nodes, 1)
        
    Returns:
        tuple: (rse, rae)
    """
    # Flatten everything to 1D vectors for easy summation over the entire test set
    preds_flat = all_preds.view(-1)
    targets_flat = all_targets.view(-1)
    
    # Calculate the mean of the ground truth (used in the denominator)
    target_mean = torch.mean(targets_flat)
    
    # --- RSE (Root Relative Squared Error) ---
    # Formula: sqrt( sum(pred - true)^2 ) / sqrt( sum(true - mean)^2 )
    numerator_rse = torch.sqrt(torch.sum((preds_flat - targets_flat) ** 2))
    denominator_rse = torch.sqrt(torch.sum((targets_flat - target_mean) ** 2))
    rse = numerator_rse / (denominator_rse + 1e-7) # Add epsilon to avoid div by zero
    
    # --- RAE (Relative Absolute Error) ---
    # Formula: sum( |pred - true| ) / sum( |true - mean| )
    numerator_rae = torch.sum(torch.abs(preds_flat - targets_flat))
    denominator_rae = torch.sum(torch.abs(targets_flat - target_mean))
    rae = numerator_rae / (denominator_rae + 1e-7)
    
    return rse.item(), rae.item()

def main():
    """
    Main entry point for the training and evaluation script.
    """
    
    # --- 1. Configuration Setup ---
    
    # Initialise Config first. Load Config from Python Class
    config = PDFormerConfig()

    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Main training script for PDFormer model.")
    
    # Path to the directory with processed data files
    # parser.add_argument('--data_dir', type=str, default='../Processed_Data/Graph',
    parser.add_argument('--data_dir', type=str, default=config.processed_data_dir,
                        help="Path to the directory with processed data files.")
    
    # Overrides for hyperparameters
    parser.add_argument('--learning_rate', type=float, default=None, 
                        help="Override for learning rate.")
    parser.add_argument('--epochs', type=int, default=None, 
                        help="Override for number of epochs.")
    parser.add_argument('--batch_size', type=int, default=None, 
                        help="Override for batch size.")
    
    args = parser.parse_args()
    
    # Apply Overrides from Command Line (Using the class method)
    # Ensure PDFormerConfig has this method, or manually override below
    if hasattr(config, 'update_from_args'):
        config.update_from_args(args)
    else:
        # Fallback manual overrides
        if args.learning_rate: config.learning_rate = args.learning_rate
        if args.epochs: config.max_epoch = args.epochs
        if args.batch_size: config.batch_size = args.batch_size

    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = str(device)
    
    print(f"Loaded configuration for model: {config.model_name}")
    print(f"Using device: {device}")

    # --- 2. Weights & Biases (Optional) ---
    # wandb.init(project="Cyber-Threat-Graph", config=config.__dict__, mode="disabled") 

    # --- 3. Data Setup ---
    print("\n--- 3. Setting up DataLoaders ---")
    
    train_dataset = CyberThreatGraphDataset(data_dir=args.data_dir, split='train')
    val_dataset = CyberThreatGraphDataset(data_dir=args.data_dir, split='val')
    # test_dataset = CyberThreatGraphDataset(data_dir=args.data_dir, split='test')

    batch_feature_name = {'X': 'float', 'y': 'float'}
    collator_fn = lambda batch_list: pdformer_collate_fn(batch_list, batch_feature_name)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=getattr(config, 'num_workers', 0), # Safer default for Colab can change for natogpu
        collate_fn=collator_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 0), # Safer default for Colab can change for natogpu
        collate_fn=collator_fn
    )
    print("DataLoaders created successfully.")

    # --- 4. Model Initialisation ---
    print("\n--- 4. Initialising Model ---")
    
    # 4a. Get Static Features (Adjacency, DTW, etc.)
    static_features = train_dataset.get_static_features()
    
    # Convert NumPy arrays to Float Tensors on Device
    for key, value in static_features.items():
        if isinstance(value, np.ndarray):
            # PDFormer expects Float tensors for matrices
            static_features[key] = torch.from_numpy(value).float().to(device)
            print(f" - Moved {key} to {device} (Shape: {value.shape})")

    # 4b. Auto-detect Dimensions from Data
    # Get one sample to check shapes: (Window, Nodes, Feats)
    sample_x, sample_y = train_dataset[0] 
    
    # Using dot notation for Dataclass assignment
    config.num_nodes = sample_x.shape[1]      
    config.feature_dim = sample_x.shape[2]    
    config.output_dim = sample_y.shape[2]     
    config.input_window = sample_x.shape[0]   
    config.output_window = sample_y.shape[0]  
    
    print(f"Auto-configured Model: Nodes={config.num_nodes}, Input Dim={config.feature_dim}")


    # # --- DEBUGGING BLOCK: Inspect static_features ---
    # print("\n--- Inspecting static_features ---")
    # if static_features is None:
    #     print("static_features is None! This is the problem.")
    # else:
    #     print(f"Keys available: {list(static_features.keys())}")
    #     for key, value in static_features.items():
    #         # Check if the value is a tensor or numpy array to print its shape
    #         if hasattr(value, 'shape'):
    #             print(f"Key: '{key}' | Shape: {value.shape}")
    #         else:
    #             print(f"Key: '{key}' | Type: {type(value)} (No shape attribute)")
    # print("------------------------------------\n")
    # # ----------------------------------------------

    # if 'adj_mx' in static_features:
    #     real_node_count = static_features['adj_mx'].shape[0]
        
    #     # Use getattr() to safely read the current value (defaults to 'Unknown' if missing)
    #     current_val = getattr(config, 'num_nodes', 'Unknown')
    #     print(f"DEBUG: Overwriting config.num_nodes from {current_val} to {real_node_count}")
        
    #     # Use dot notation to set the value on the object
    #     config.num_nodes = real_node_count
        
    #     # Update 'num_vertex' if the object has that attribute
    #     if hasattr(config, 'num_vertex'):
    #          config.num_vertex = real_node_count
    # # -----------------------------------------------------

    # 4c. Initialise Model
    model = PDFormer_Wrapper(model_config=config, data_feature=static_features).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialised. Total trainable parameters: {total_params}")

    # --- 5. Training Components ---
    print("\n--- 5. Setting up Loss and Optimiser ---")
    
    loss_fn = nn.L1Loss() # MAE Loss
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epoch)
    
    # --- 6. Training Loop ---
    print("\n--- 6. Starting Training ---")
    
    # Path to save the best model
    # We save it in the 'Models' folder inside the pipeline
    save_path = os.path.join(SCRIPT_DIR, 'Models', 'best_model_graph.pth')
    
    best_val_loss = float('inf') # Track the best score
    
    for epoch in range(config.max_epoch):
        start_time = time.time()
        model.train()
        train_epoch_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            batch.to_tensor(device)
            y_true = batch['y'] 
            
            optimizer.zero_grad()
            y_predicted = model(batch)
            loss = loss_fn(y_predicted, y_true)
            
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
        
        avg_train_loss = train_epoch_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        val_epoch_loss = 0.0
        val_preds_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch.to_tensor(device)
                y_true = batch['y']
                y_predicted = model(batch)
                
                loss = loss_fn(y_predicted, y_true)
                val_epoch_loss += loss.item()
                
                val_preds_list.append(y_predicted.cpu())
                val_targets_list.append(y_true.cpu())

        avg_val_loss = val_epoch_loss / len(val_loader)
        
        # Calculate Paper Metrics
        all_val_preds = torch.cat(val_preds_list, dim=0)
        all_val_targets = torch.cat(val_targets_list, dim=0)
        val_rse, val_rae = compute_metrics(all_val_preds, all_val_targets)
        
        scheduler.step()
        
        # --- SAVE MODEL IF BEST ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the state dictionary (weights only)
            torch.save(model.state_dict(), save_path)
            save_msg = f"💾 Saved New Best Model (Loss: {best_val_loss:.4f})"
        else:
            save_msg = ""
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.max_epoch} | "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"RSE: {val_rse:.4f} | "
              f"{save_msg}")

    print(f"\n--- Training Complete. Best Model Saved to: {save_path} ---")

if __name__ == '__main__':
    main()