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
from Graph_Dataset import CyberThreatGraphDataset
# Ensure you have this wrapper available
from Models.PDFormer_Wrapper import PDFormerModel 
# Ensure this path matches your library structure
from Transformers.Graph_Transformer.libcity.data.batch import Batch 

def pdformer_collate_fn(batch_list, feature_name):
    """
    Custom collator: Converts list of NumPy samples into a libcity Batch object.
    """
    batch = Batch(feature_name)
    for x_sample, y_sample in batch_list:
        batch.append((x_sample, y_sample))
    return batch

def main():
    """
    Main entry point for the training and evaluation script.
    """
    
    # --- 1. Configuration Setup ---
    print("--- 1. Setting up Configuration ---")
    parser = argparse.ArgumentParser(description="Main training script for PDFormer model.")
    
    parser.add_argument('--config_file', type=str, default='pdformer_config.json',
                        help="Path to the JSON config file.")
    # UPDATED DEFAULT PATH to match our pipeline
    parser.add_argument('--data_dir', type=str, default='../Processed_Data/graph',
                        help="Path to the directory with processed data files.")
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help="Override for learning rate.")
    parser.add_argument('--epochs', type=int, default=10, 
                        help="Override for number of epochs.")
    parser.add_argument('--batch_size', type=int, default=16, 
                        help="Override for batch size.")
    
    args = parser.parse_args()

    # Load Config
    config_path = os.path.join(SCRIPT_DIR, args.config_file)
    # Basic default config in case JSON is missing
    config = {
        'learning_rate': 0.001, 
        'max_epoch': 10, 
        'batch_size': 16, 
        'weight_decay': 0.0001,
        'embed_dim': 64, 
        'skip_depth': 2, 
        'lape_dim': 8, 
        'geo_num_heads': 4, 
        'sem_num_heads': 2,
        'num_workers': 0
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
            print(f"Loaded configuration from {config_path}")
    else:
        print(f"Warning: Config file {config_path} not found. Using defaults.")

    # Overrides
    if args.learning_rate: config['learning_rate'] = args.learning_rate
    if args.epochs: config['max_epoch'] = args.epochs
    if args.batch_size: config['batch_size'] = args.batch_size

    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = str(device)
    print(f"Using device: {device}")

    # --- 2. Weights & Biases (Optional) ---
    # wandb.init(project="Cyber-Threat-Graph", config=config, mode="disabled") 

    # --- 3. Data Setup ---
    print("\n--- 3. Setting up DataLoaders ---")
    
    train_dataset = CyberThreatGraphDataset(data_dir=args.data_dir, split='train')
    val_dataset = CyberThreatGraphDataset(data_dir=args.data_dir, split='val')
    # test_dataset = CyberThreatGraphDataset(data_dir=args.data_dir, split='test')

    batch_feature_name = {'X': 'float', 'y': 'float'}
    collator_fn = lambda batch_list: pdformer_collate_fn(batch_list, batch_feature_name)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        collate_fn=collator_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        collate_fn=collator_fn
    )
    print("DataLoaders created successfully.")

    # --- 4. Model Initialisation ---
    print("\n--- 4. Initialising Model ---")
    
    # 4a. Get Static Features (Adjacency, DTW, etc.)
    static_features = train_dataset.get_static_features()
    
    # CRITICAL UPDATE: Convert NumPy arrays to Float Tensors on Device
    for key, value in static_features.items():
        if isinstance(value, np.ndarray):
            # PDFormer expects Float tensors for matrices
            static_features[key] = torch.from_numpy(value).float().to(device)
            print(f" - Moved {key} to {device} (Shape: {value.shape})")

    # 4b. Auto-detect Dimensions from Data
    # Get one sample to check shapes: (Window, Nodes, Feats)
    sample_x, sample_y = train_dataset[0] 
    
    config['num_nodes'] = sample_x.shape[1]      # Auto-set number of nodes
    config['feature_dim'] = sample_x.shape[2]    # Auto-set input features (usually 1)
    config['output_dim'] = sample_y.shape[2]     # Output features (usually 1)
    config['input_window'] = sample_x.shape[0]   # Auto-set window size
    config['output_window'] = sample_y.shape[0]  # Auto-set horizon
    
    print(f"Auto-configured Model: Nodes={config['num_nodes']}, Input Dim={config['feature_dim']}")

    # 4c. Initialise Model
    model = PDFormerModel(model_config=config, data_feature=static_features).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialised. Total trainable parameters: {total_params}")

    # --- 5. Training Components ---
    print("\n--- 5. Setting up Loss and Optimiser ---")
    
    loss_fn = nn.L1Loss() # MAE Loss
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epoch'])
    
    # --- 6. Training Loop ---
    print("\n--- 6. Starting Training ---")
    
    for epoch in range(config['max_epoch']):
        start_time = time.time()
        model.train()
        train_epoch_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            # Move Batch to device (libcity Batch object handles this)
            batch.to_tensor(device)
            
            # y_true shape: (batch, time, nodes, feats)
            y_true = batch['y'] 
            
            optimizer.zero_grad()
            
            # Forward Pass
            y_predicted = model(batch)
            
            # Calculate Loss
            loss = loss_fn(y_predicted, y_true)
            
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
        
        avg_train_loss = train_epoch_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        val_epoch_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch.to_tensor(device)
                y_true = batch['y']
                y_predicted = model(batch)
                
                loss = loss_fn(y_predicted, y_true)
                val_epoch_loss += loss.item()

        avg_val_loss = val_epoch_loss / len(val_loader)
        scheduler.step()
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config['max_epoch']} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Time: {epoch_time:.1f}s")
        
        # wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss})

    print("\n--- Training Complete ---")

if __name__ == '__main__':
    main()