# Preprocessing Pipeline

This directory contains all scripts related to the **offline preprocessing** of the Cyber Trend Forecasting dataset. he purpose of these scripts is to transform the single, raw `Cyber_Trend_Forecasting_All.csv` file into a collection of processed output files that are in format our Graph Transformer require for training.

## Overview of the Transformation

The transformation process converts the single, flat CSV file (which is a 2D table of *time* vs. *threats*) into a complete, graph-based dataset ready for spatio-temporal forecasting.

First, we define the **spatial structure** (the "map"). We treat each of the 113 threat columns as a "node" in a network. We then calculate the statistical correlation between every pair of nodes (using the training data) to create our primary **adjacency matrix**, which defines the connections of the graph.

Second, we process the **temporal features**. We take the raw daily trend values for all 113 nodes, normalize them (so all values are between 0 and 1), and then use a sliding window to create "samples." Each sample consists of an input "chunk" (e.g., 30 days of data) and a corresponding target "chunk" (e.g., the 1 day we want to predict). These windowed `(X, y)` pairs are saved as our core training, validation, and test data.

Finally, to support the specific needs of the PDFormer model, we run several *optional*, additional one-time computations. This includes calculating a **Dynamic Time Warping (DTW) matrix** to find nodes with similar-shaped trends, computing **shortest-path matrices** to understand how threats propagate, and running **time-series clustering** to identify common patterns. All of these -the windowed data, the adjacency matrix, the DTW matrix, and the cluster keys - are saved as separate files.

## Scripts in this Directory

* **`Load_Data.py`**:
    * Provides a single, reusable function (`load_cyber_threat_data`) to load the raw `Cyber_Trend_Forecasting_All.csv` file.
    * Handles date parsing and sets the 'Date' column as the DataFrame index.
    * Performs basic validation (checking file existence, sorting data).

* **`Cyber_Trend_to_Graph.py`**:
    * This is the main offline preprocessing engine for all graph models.
    * Imports the loading function from `Load_Data.py`.
    * Runs the **Generic Pathway** (default): Creates the base dataset, including `train.npz`, `val.npz`, `test.npz`, `adj_mx.npy`, and `scaler.pkl`.
    * Runs the **Extended Pathway** (optional, via `--pdformer` flag): Computes and saves the additional, computationally expensive artifacts required by the PDFormer model (e.g., `dtw_matrix.npy`, `sh_mx.npy`, `pattern_keys.npy`).

## How to Run

To generate the data, navigate to this directory (`Transformer_Pipeline/Preprocessing/`) and run the `Cyber_Trend_to_Graph.py` script from your terminal.

**1. Generic Pathway (Fast for a base Graph Transformer):**
This is the default mode. It generates the base dataset.
```bash
python Cyber_Trend_to_Graph.py
```
This will generate: `train.npz`, `val.npz`, `test.npz`, `adj_mx.npy`, and `scaler.pkl` in the data/processed_graph/ directory.

**2. Extended Pathway (Slow, for PDFormer):** This mode runs the generic pathway and the additional, slow computations required for the PDFormer model. **NOTE: The pdformer args required to run**
```bash
python Cyber_Trend_to_Graph.py --pdformer
```

This will generate all the basefiles plus: ```dtw_matrix.npy```, ```sh_mx.npy```, ```sd_mx.npy```and ```pattern_keys.npy```