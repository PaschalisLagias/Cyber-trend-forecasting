# Data Pipeline & B-MTGNN Architecture Components

This directory contains the scripts responsible for data ingestion, preprocessing, training, evaluation, and visual output generation for the B-MTGNN architecture and unified comparative visualisations.

## Core B-MTGNN Training & Ingestion Scripts (found in ../B-MTGNN)

* **`train_test.py` (Hyperparameter Optimisation)**
  Executes a random grid search across a predefined parameter space (e.g., learning rate, graph convolution depth, dropout). It evaluates models using a sliding window approach over the validation/test sets and exports the highest-performing configuration to `model/Bayesian/hp.txt`.

* **`train.py` (Final Model Training)**
  The primary execution script for model deployment. It reads the optimal parameters defined in `hp.txt`, instantiates the Graph Neural Network, and trains the model. The final model weights are saved to a `.pt` file (e.g., `o_model.pt`) for future inference.

* **`util.py` (Standard Data Loader)**
  Handles data normalisation, batching, and graph generation during the testing and optimisation phase. It reads the raw text array and adjacency matrix, calculates symmetric/asymmetric normalised Laplacians, and partitions the dataset into strict train, validation, and test splits to prevent data leakage.

* **`o_util.py` (Deployment Data Loader)**
  A variant of the standard data loader used exclusively for final forecasting. It bypasses the train/validation/test split logic, instead feeding 100% of the historical data into the training loop. This ensures the final model has the most recent temporal context available before generating forward-looking predictions.

## Pipeline Helper & Visualisation Utilities

* **`BMTGNN_Visualise_Wrapper.py`**
  Invokes visual output subroutines for B-MTGNN forecast arrays and formats axes for output plot exports.

* **`Evaluation_Metrics.py`**
  Calculates multi-horizon error metrics (RSE, RAE, MAE, Pearson correlation) for model predictions against historical ground truth.

* **`Forecast_Export.py`**
  Exports 36-month prediction arrays into structured CSV format for external analysis.

* **`Generate_Clean_Evaluation_Tables.py`**
  Evaluates model prediction arrays across isolated target threat vectors using micro-averaged metrics and renders formatted PNG tables.

* **`Generate_Column_Mapping.py`**
  Parses raw dataset headers and builds standardised column index mappings matching feature strings to target threat labels.

* **`Prep_Training_Holdout.py`**
  Partitions historical time-series datasets into designated training and validation holdout subsets.

* **`Prep_Unsmoothed_Data.py`**
  Ingests raw tabular time series, formats text-based input arrays, and constructs the graph adjacency matrix for B-MTGNN ingestion.

* **`Unified_Visualise_Comparison.py`**
  Generates side-by-side comparison plots comparing B-MTGNN and VisionTS++ 36-month trend forecasts against mitigation technologies.

* **`Unified_Visualise_Figure3.py`**
  Generates Figure 3 validation plots comparing historical ground truth against smoothed model predictions across target threats.

* **`Visualise_Comparison.py`**
  Generates individual comparison graphics and gap analysis plots for model forecast outputs.

* **`column_mapping.csv`**
  CSV lookup file mapping raw column header strings to canonical target threat vector names.