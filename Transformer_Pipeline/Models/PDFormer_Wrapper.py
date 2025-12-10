# Standard library imports
import sys
import os
from dataclasses import asdict

# Third-party imports
import torch.nn as nn

# --- Add Paths to System ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# 1. Add Project Root (for standard imports)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 2. Add 'Transformers/Graph_Transformer' (The Submodule Root)
# Allows the PDFormer files to run "from libcity.model..." without crashing.
submodule_root = os.path.join(PROJECT_ROOT, 'Transformers', 'Graph_Transformer')
if submodule_root not in sys.path:
    sys.path.append(submodule_root)
# --- End Path Setup ---


# Local imports
# import the original PDFormer from the submodule
try:
    from Transformers.Graph_Transformer.libcity.model.traffic_flow_prediction.PDFormer import PDFormer
except ImportError as e:
    print(f"Error: Could not import PDFormer from submodule.")
    print(f"Debug: Tried to find it in: {submodule_root}")
    print("Please ensure the submodule is initialised: `git submodule update --init --recursive`")
    raise e


class PDFormerModel(nn.Module):
    """
    A PyTorch nn.Module wrapper for the PDFormer model.

    This class acts as a clean, standard interface between our training
    pipeline and the complex, config-driven PDFormer model which lives in
    the submodule.

    It handles the complex initialisation of the PDFormer "engine" and 
    ensures its forward pass is compatible with a standard PyTorch training loop.

    --- IMPORTANT: PREDICTION VS. LOSS ---
    The original PDFormer class has its own `.calculate_loss()` 
    method. This wrapper class intentionally **IGNORES** that method.

    Our `forward()` method (below) is designed to call the class's 
    `.predict()` method, which returns only the raw prediction tensor.
    We then calculate the loss **externally** in our `Cyber_Trend_Train.py` 
    script using standard PyTorch functions (nn.MSELoss() & nn.L1Loss()),
    which is a cleaner, more modular, and more flexible design.
    """

    def __init__(self, model_config, data_feature):
        """
        Initialises the PDFormer wrapper and the underlying engine.

        Args:
            model_config (dict): A dictionary of model hyperparameters 
                                 (e.g., embed_dim, enc_depth, input_window). 
                                 This is the 'config' object from the original framework,
                                 which we will build in our Train.py script.
            data_feature (dict): A dictionary of all static data artifacts 
                                 (e.g., adj_mx, dtw_matrix, sh_mx, scaler, num_nodes) 
                                 loaded by Graph_Dataset.get_static_features().
        """
        super().__init__()

        # --- COMPATIBILITY FIX ---
        # The 'libcity' library expects 'config' to be a Dictionary (so it can use .get()).
        # Our pipeline uses a modern Dataclass. We must convert it here.
        if hasattr(model_config, '__dataclass_fields__'):
            config_dict = asdict(model_config)
        elif isinstance(model_config, dict):
            config_dict = model_config
        else:
            # Fallback for other object types
            config_dict = model_config.__dict__

        # -------------------------------------------------------------
        # --- SAFETY CHECK: Force num_nodes to match Data Shape ---
        # -------------------------------------------------------------
        # This prevents the [1,1] vs [1231,1231] mismatch error by ensuring
        # the config dictionary matches the actual adjacency matrix.
        if 'adj_mx' in data_feature:
            real_node_count = data_feature['adj_mx'].shape[0]
            
            # 1. Update Config
            config_dict['num_nodes'] = real_node_count
            config_dict['num_vertex'] = real_node_count
            
            # 2. Update Data Feature (Some models look here instead of config)
            data_feature['num_nodes'] = real_node_count
            
            print(f"Wrapper: Forced config num_nodes to {real_node_count}")
        else:
            print("Wrapper WARNING: 'adj_mx' not found in data_feature!")
        # -------------------------------------------------------------    

        # -------------------------------------------------------------
        # --- TYPE FIX: Convert pattern_keys to Numpy ---
        # -------------------------------------------------------------
        # The library 'PDFormer.py' (line 382) explicitly calls torch.from_numpy(),
        # so it will crash if we feed it a Tensor. We must downgrade it to Numpy here.
        if 'pattern_keys' in data_feature:
            pk = data_feature['pattern_keys']
            # Check if it is a Tensor (using string check avoids importing torch if not needed)
            if 'Tensor' in str(type(pk)): 
                data_feature['pattern_keys'] = pk.detach().cpu().numpy()
                print("Wrapper: Converted pattern_keys from Tensor to Numpy for compatibility.")


        # -------------------------------------------------------------
        # --- SHAPE FIX: Update s_attn_size from pattern_keys ---
        # -------------------------------------------------------------
        if 'pattern_keys' in data_feature:
            pk = data_feature['pattern_keys']
            # pk shape is likely (num_patterns, vector_dim, output_dim)
            # e.g., (16, 10, 1). The middle dimension (10) is the s_attn_size.
            
            # Use .shape if it's a tensor/array, otherwise assume list
            pk_shape = pk.shape if hasattr(pk, 'shape') else np.array(pk).shape
            
            # The dimension mismatch was 10 vs 3. 10 is at index 1 of the shape [16, 10, 1]
            real_s_attn_size = pk_shape[1]
            
            # Check if config needs updating
            current_s_attn = config_dict.get('s_attn_size', 'Unknown')
            
            if current_s_attn != real_s_attn_size:
                print(f"Wrapper: Overriding s_attn_size from {current_s_attn} to {real_s_attn_size}")
                config_dict['s_attn_size'] = real_s_attn_size
        # -------------------------------------------------------------


        # -------------------------------------------------------------
        # --- FEATURE FIX: Disable Time/Day Embeddings ---
        # -------------------------------------------------------------
        # The model defaults to expecting input shape [B, T, N, feature_dim + 2].
        # It tries to read "Time of Day" at index [feature_dim].
        # Your data is [B, T, N, 1], so index 1 doesn't exist.
        # We must disable these flags to prevent the IndexError.
        
        # Check if feature_dim is 1 (or default to 1)
        f_dim = config_dict.get('feature_dim', 1)
        
        # If we have 1 feature, we almost certainly lack the extra time channels
        # (unless feature_dim was explicitly set to include them, which is rare).
        if f_dim == 1:
            print("Wrapper: Disabling 'add_time_in_day' and 'add_day_in_week' (Data lacks time channels).")
            config_dict['add_time_in_day'] = False
            config_dict['add_day_in_week'] = False
        # -------------------------------------------------------------

        # Instantiate the original PDFormer model
        self.pdformer_engine = PDFormer(config=config_dict, 
                                        data_feature=data_feature)
                
        # Instantiate the original PDFormer model from the submodule
        # We pass it the config and the dictionary of pre-computed data features.
        self.pdformer_engine = PDFormer(config=config_dict, 
                                        data_feature=data_feature)
        print("PDFormer engine initialised successfully.")

    def forward(self, batch):
        """
        Defines the forward pass of the model.

        This method calls the original PDFormer's `.predict()` method 
        to get the raw output predictions.

        Args:
            batch (libcity.data.batch.Batch): The custom Batch object 
                provided by our DataLoader's collator function. 
                This object contains the `batch['X']` data.

        Returns:
            torch.Tensor: The raw prediction tensor from the model.
        """
        # Call the classes predict method.
        # The .predict() method in the original code is just a wrapper
        # around its own .forward() method.
        # This returns the final prediction tensor.
        y_predicted = self.pdformer_engine.predict(batch)
        
        return y_predicted