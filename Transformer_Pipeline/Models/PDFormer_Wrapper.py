# Standard library imports
import sys
import os
from dataclasses import asdict

# Third-party imports
import torch
import torch.nn as nn
import numpy as np

# --- Add Paths to System ---
# Get the directory of the current script (Transformer_Pipeline/Models)
if "__file__" in globals():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    SCRIPT_DIR = os.getcwd()

# Project Root is two levels up (../../)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Add the Graph Transformer submodule to path
submodule_root = os.path.join(PROJECT_ROOT, 'Transformers', 'Graph_Transformer')
if submodule_root not in sys.path:
    sys.path.append(submodule_root)
# --- End Path Setup ---

# Local imports
try:
    from Transformers.Graph_Transformer.libcity.model.traffic_flow_prediction.PDFormer import PDFormer
except ImportError as e:
    print(f"Error: Could not import PDFormer from submodule.")
    raise e


class PDFormer_Wrapper(nn.Module):
    """
    A PyTorch Wrapper for the PDFormer model.
    It handles Config translation, Input reshaping, and critical bug fixes.
    """

    def __init__(self, model_config, data_feature):
        """
        Initialises the PDFormer wrapper and the underlying engine.
        """
        super().__init__()

        # --- 1. CONFIG PREPARATION ---
        if hasattr(model_config, '__dataclass_fields__'):
            config_dict = asdict(model_config)
        elif isinstance(model_config, dict):
            config_dict = model_config
        else:
            config_dict = model_config.__dict__

        # --- 2. CONFIG FIXES ---
        # Ensure num_nodes matches the actual data
        if 'adj_mx' in data_feature:
            real_node_count = data_feature['adj_mx'].shape[0]
            config_dict['num_nodes'] = real_node_count
            config_dict['num_vertex'] = real_node_count
            data_feature['num_nodes'] = real_node_count

        # Fix Feature Flags
        f_dim = config_dict.get('feature_dim', 1)
        if f_dim == 1:
            config_dict['add_time_in_day'] = False
            config_dict['add_day_in_week'] = False

        # --- 3. PATTERN KEYS FIX (CRITICAL TYPE CAST) ---
        if 'pattern_keys' in data_feature:
            pk = data_feature['pattern_keys']
            # Ensure it is numpy
            if 'Tensor' in str(type(pk)): 
                pk = pk.detach().cpu().numpy()
            
            # FORCE FLOAT32 to avoid "Double vs Float" errors
            data_feature['pattern_keys'] = pk.astype(np.float32)

            # Fix s_attn_size based on pattern_keys shape
            pk_shape = pk.shape
            real_s_attn_size = pk_shape[1]
            config_dict['s_attn_size'] = real_s_attn_size

        # --- 4. LAPLACIAN MATRIX SETUP ---
        if 'lap_mx' in data_feature:
            lap_np = data_feature['lap_mx']
            
            # Register as buffer (Moves to GPU automatically)
            # Ensure FLOAT32 here as well
            self.register_buffer(
                'lap_mx', 
                torch.tensor(lap_np, dtype=torch.float32)
            )
            self.needed_lap_dim = lap_np.shape[1]
        else:
            self.lap_mx = None
            self.needed_lap_dim = 8  # Default fallback

        # --- 5. INSTANTIATE ENGINE ---
        self.pdformer_engine = PDFormer(config=config_dict, 
                                        data_feature=data_feature)

        # --- 6. EXECUTE "SEARCH AND DESTROY" PATCH ---
        if self.lap_mx is not None:
            self._patch_model_layers(self.pdformer_engine, self.needed_lap_dim)

    def _patch_model_layers(self, module, needed_dim):
        """
        Recursively searches for 'embedding_lap_pos_enc' and replaces it 
        if the dimension doesn't match the data.
        """
        did_patch = False
        for name, child in module.named_modules():
            if "embedding_lap_pos_enc" in name and isinstance(child, nn.Linear):
                current_dim = child.in_features
                if current_dim != needed_dim:
                    # Create the correct layer
                    new_layer = nn.Linear(needed_dim, child.out_features)
                    new_layer.to(child.weight.device) 
                    self._set_module_by_name(module, name, new_layer)
                    did_patch = True
        
        if did_patch:
            # print("Wrapper: Patched Laplacian Embedding Layer dimensions.")
            pass

    def _set_module_by_name(self, base_module, name, new_module):
        levels = name.split('.')
        parent = base_module
        for level in levels[:-1]:
            parent = getattr(parent, level)
        setattr(parent, levels[-1], new_module)

    def forward(self, batch):
        """
        Forward pass adapter.
        Handles both Dict inputs (Train) and Tensor inputs (Eval).
        """
        # --- ADAPTER FIX ---
        # 1. If input is raw Tensor, wrap in Dict
        if isinstance(batch, torch.Tensor):
            batch = {'X': batch}
        
        # 2. CRITICAL: Enforce Float32 on input data
        # Data loaders often output Float64 (Double) which crashes the model
        if 'X' in batch and batch['X'].dtype == torch.float64:
            batch['X'] = batch['X'].float()

        # Pass the Laplacian buffer explicitly
        y_predicted = self.pdformer_engine.predict(batch, self.lap_mx)
        return y_predicted