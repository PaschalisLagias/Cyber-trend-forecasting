# Transformer Pipeline

This directory contains all the original source code for our custom transformer-based experiments. It includes the scripts for data preprocessing, the PyTorch `Dataset` class, the model definitions, and the main script for training and evaluating the models.

This code is kept separate from the original author's forked codebase (in the root directory) and from the third-party model submodules (in the `/Transformers` directory).

## Experiment Overview

This project approaches the forecasting problem using two different models to benchmark against the B-MTGNN baseline:

1.  **Graph Transformer (PDFormer)**: Models the cyber threat landscape as a spatio-temporal graph.
2.  **Vision Transformer (VisionTS)**: Converts multivariate time-series data into 2D images using Colourised Multivariate Conversion (CMC) to leverage a Masked Autoencoder (MAE) backbone.

## Directory Structure
```
Transformer_Pipeline/
├── Models/
│   └── PDFormer_Wrapper.py
├── Checkpoints/  # Stores the saved model weights (`.pth` files) generated during training. The pipeline automatically saves the "best" model based on validation loss here
├── Preprocessing/
│   ├── Load_Data.py
│   ├── Cyber_Trend_to_Graph.py
│   └── Cyber_Trend_to_Image.py
├── Results/ # Stores .npy files, .csv tables, and .png plots
│   ├── predictions.npy: Raw model forecast tensors.
│   ├── ground_truth.npy: The actual target values used for comparison.
│   └── plots_paper_style: The sub-folder containing all generated PNGs and LaTeX tables (Figs 3 & 4, Tables 5 & 6).
├── Utils/
│   └── Metrics.py
├── Cyber_Trend_Graph_Dataset.py
├── Cyber_Trend_Graphy_Config.json
├── Cyber_Trend_Vision_Dataset.py
├── Cyber_Trend_Vision_Config.json
├── Evaluate_Graph.py
├── Train_Graph.py
├── Train_Vision.py
├── requirements.tx
├── Run_Pipeline.py
└── Visualise_Results.py
```

## Usage

1.  **Configure:** Edit `Cyber_Trend_Graph_Config.py` to set your desired hyperparameters (batch size, learning rate, etc.).
2.  **Train:** Run `python Train_Graph.py` to train the model.
3.  **Evaluate:** Run `python Evaluate_Graph.py` to generate predictions.
4.  **Visualise:** Run `python Visualise_Results.py` to create graphs and tables.

## Experimental Settings & Metrics

To ensure a fair comparison with the benchmark paper (*Forecasting Cyber Threats and Pertinent Mitigation Technologies*), pipelines has following settings:

* **Input Window**: 10 Months
* **Forecast Horizon**: 36 Months
* **Data Split**: 43% Training, 30% Validation, 27% Testing

**Evaluation Metrics**
Performance is evaluated using the global metrics defined in the paper (calculated over the entire test set):
* **RSE (Root Relative Squared Error)**: $$sqrt(sum(pred - true)^2) / sqrt(sum(true - mean)^2)$$
* **RAE (Relative Absolute Error)**: $$sum(|pred - true|) / sum(|true - mean|)$$

## Components

Here is a high-level overview of the files and folders in this pipeline:

### Preprocessing Scripts

* **`Preprocessing/` (Sub-directory)**
    * **Purpose:** This folder contains all **offline** scripts used to convert the raw `.csv` data into the processed files required for training.
    * **Note:** This directory has its own `README.md` with detailed instructions on how to run the preprocessing scripts and what the optional flags (like `--pdformer`) do.
    * **Key Scripts**:
        * `Load_Data.py`: Handles the specific `MMM-YY` date format parsing.
        * `Cyber_Trend_to_Graph.py`: Applies Double Exponential Smoothing (DES) and generates graph tensors (Graph Pipeline).
        * `Cyber_Trend_to_Image.py`: Applies DES smoothing, creates sliding windows (no external normalisation).

### Dataset Script

The `Cyber_Trend_Graph_Dataset.py` script contains the core PyTorch Dataset class for the graph project. The class links the processed data files and the DataLoader.The `Cyber_Trend_Vision_Dataset.py` script contains ...

* **`Cyber_Trend_Graph_Dataset.py`**
    * **Purpose**: Defines a custom PyTorch `Dataset` class (`CyberThreatGraphDataset`) to interface between processed files and the training loop.
    * **Process**:
        * Loads the pre-processed, windowed data (`train.npz`) for a specific split.
        * `get_static_features()`: Loads and returns the dictionary of static artifacts (`adj_mx`, `dtw_matrix`) needed by the model constructor.
* **`Cyber_Trend_Image_Dataset.py`**
    * **Purpose**:  PyTorch Dataset for the Vision pipeline following `Cyber_Trend_Graph_Dataset.py`.
    * **Process**:
        *  Loads pre-processed `.npz` files.
        * `get_static_features()`: Returns scaler and metadata for reference (not used by model).
        * Returns `{"input": tensor, "target": tensor}` dictionaries with 3D tensors.


### Model Scripts

The Model subdirectory contains model architecture files, i.e., the wrapper for PDFormer.

* **`Models/` (Sub-directory)**
    * **Purpose**: Contains custom-written model architecture files.
    * **Content**:
        * `PDFormer_Wrapper.py` defines the `PDFormerModel` class, which wraps the complex internal engine and exposes a clean `forward()` method.
        * **`Models/VisionTS_Wrapper.py`** Wraps the VisionTS model from `Transformers/Visual_Transformer/` submodule.
            * Handles `update_config()` for context/prediction length settings.
            * VisionTS performs normalisation/denormalisation internally.

### Training Script

Main training scripts

* **`Train_Graph.py`**
    * **Purpose**: The executable script to run the entire training and evaluation experiment.
    * **Process**:
        1.  Imports the `Batch` class from the `libcity` submodule.
        2.  Initialises `CyberThreatGraphDataset` and the `PDFormer` model.
        3.  Runs the optimisation loop.
        4.  Calculates global RSE/RAE metrics at the end of each epoch.
* **`Train_Vision.py`**
    * **Purpose**: Training and evaluation for VisionTS model.
    * **Process**:
        1.  ...
        2.  ...


### Evaluation Scripts

* **`Utils/` (Sub-directory)**
    * **Purpose**: Acts as the shared mathematical library for the project, for consistency across training logs and final paper results.
    * **Key Scripts**:
        * `Metrics.py`: Defines the core evaluation formulas: RSE (Root Relative Squared Error) and RAE (Relative Absolute Error) for B-MTGNN benchmarking. `calculate_gap()`: Performs vector subtraction between Threat and Mitigation arrays to derive the "Risk Gap" for downstream analysis.


* **`Evaluate_Graph.py`**
    * **Purpose**: The post-training analysis script that generates the quantitative results tables and raw prediction artifacts for write-ups.
    * **Process**:
        * Loads the trained `best_model_graph.pth` and runs inference on the Test Set.
        * Performs Inverse Scaling to convert normalised model outputs back into real-world values (e.g., attack counts)
        * Calculates metrics (RSE, RAE, MAE) across specific horizons (3, 6, 12, 24 months) and exports them to `graph_evaluation_results.csv`.
        * Saves raw predictions (`.npy`) and column metadata (`.json`) to the Results/ folder for visualisation.

* **`Visualise_Results.py`**
    * **Purpose**: A dedicated visualisation script that transforms raw prediction data into high-resolution, paper-ready figures without re-running the model.
    * **Process**:
        * Loads the raw prediction and ground truth files (`.npy`) generated by the evaluation step..
        * `get_col_index()`: Dynamically locates specific threats (e.g., "DDoS-ALL") using the exported metadata.
        * Generates the Gap Analysis plot (shading the region where Threat > Mitigation) and Forecast Verification plots for the final report.


### Pipeline Script

Main Pipeline script

* **`Run_Pipeline.py`**
    * **Purpose**: The single, main entry point for all experiments. Handles path setup and conditional execution.
    * **Process**:
        * Parses user arguments (`--model`, `--force-preprocess`).
        * Checks if processed data files exist.
        * Conditionally runs the correct preprocessing module (Graph or Vision).
        * Conditionally runs the correct training script.

## How to Run the Pipeline

**Verification (Recommended First Step)**
To visualise data smoothing and verify tensor shapes interactively, run the notebook: `../Notebooks/Graph_Pipeline_Verification.ipynb`.

**1. Generate the Data:**
Run the preprocessing module from the project root:

```bash
python -m Transformer_Pipeline.Preprocessing.Cyber_Trend_to_Graph --pdformer
```

**2. Training Model and Log Results:**

Execute the pipeline script:

```bash
python Transformer_Pipeline/Run_Pipeline.py --model graph
```

## Dependencies

All Python libraries required to run the code in this pipeline are listed in the `requirements.txt` file at the root of the main project repository. Please ensure you have installed them before running any scripts.
