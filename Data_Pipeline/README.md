### B-MTGNN Core Scripts Overview

The B-MTGNN architecture relies on four primary Python scripts to manage data ingestion, hyperparameter optimisation, and model training. 

*   **`train_test.py` (Hyperparameter Optimisation)**
    Executes a random grid search across a predefined parameter space (e.g., learning rate, graph convolution depth, dropout). It evaluates models using a sliding window approach over the validation/test sets and exports the highest-performing configuration to `model/Bayesian/hp.txt`.

*   **`train.py` (Final Model Training)**
    The primary execution script for model deployment. It reads the optimal parameters defined in `hp.txt`, instantiates the Graph Neural Network, and trains the model. The final model weights are saved to a `.pt` file (e.g., `o_model.pt`) for future inference.

*   **`util.py` (Standard Data Loader)**
    Handles data normalisation, batching, and graph generation during the testing and optimisation phase. It reads the raw text array and adjacency matrix, calculates symmetric/asymmetric normalised Laplacians, and partitions the dataset into strict train, validation, and test splits to prevent data leakage.

*   **`o_util.py` (Deployment Data Loader)**
    A variant of the standard data loader used exclusively for final forecasting. It bypasses the train/validation/test split logic, instead feeding 100% of the historical data into the training loop. This ensures the final model has the most recent temporal context available before generating forward-looking predictions.