# Standard library imports
import sys
import os

# Third-party imports
import torch
import torch.nn as nn

# --- Add Project Root to Python Path ---
# This is a standard way to allow this script (in Models/) to 
# import from the 'Transformers/' directory at the project root.
# Assumes 'Cyber_Trend_Train.py' (which imports this) will be run from 'Transformer_Pipeline/'.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# --- End Path Setup ---

# Local imports
# Now we can import the original PDFormer "engine" from the submodule
try:
    from Transformers.Graph_Transformer.libcity.model.pdformer import PDFormer
except ImportError as e:
    print(f"Error: Could not import PDFormer from submodule.")
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

        # Instantiate the original PDFormer model from the submodule
        # We pass it the config and the dictionary of pre-computed data features.
        self.pdformer_engine = PDFormer(config=model_config, 
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