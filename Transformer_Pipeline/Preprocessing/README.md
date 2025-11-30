# Preprocessing Pipeline

This directory contains the scripts for the **offline preprocessing** of the Cyber Trend Forecasting dataset. These scripts transform the raw `Cyber_Trend_Forecasting_All.csv` file into the processed 4D tensors required for training.

## Overview of the Graph Transformation

The transformation process converts the flat CSV file (a 2D table of *Time* vs. *Threats*) into a complete, graph-based dataset optimized for spatio-temporal forecasting.

### 1. Data Ingestion & Smoothing
We load the raw data, explicitly parsing the **`MMM-YY`** date format to ensure the time-series is sorted chronologically. We then apply **Double Exponential Smoothing (DES)** ($\alpha=0.1, \beta=0.1$) to reduce noise and capture the underlying trend, aligning with the B-MTGNN benchmark methodology.

### 2. Spatial Structure (The Graph)
We treat each of the threat columns as a "node" in a network. We calculate the statistical correlation between every pair of nodes (using the smoothed training data) to create the primary **Adjacency Matrix** (`adj_mx.npy`), which defines the graph topology.

### 3. Temporal Features (Windowing)
We normalise the smoothed trend values (0 to 1) and slice the time-series into "samples" using the strict experimental settings defined in the benchmark paper:
* **Input Window:** 10 Months (History)
* **Forecast Horizon:** 36 Months (Future Target)
* **Splits:** 43% Train, 30% Validation, 27% Test

### 4. PDFormer Data (Extended Pathway)
To support the PDFormer architecture, we run additional computations:
* **DTW Matrix:** Dynamic Time Warping distances between nodes.
* **Shortest Path Matrices:** Hop counts and distances based on the adjacency matrix.
* **Cluster Keys:** Representative time-series patterns (via K-Means) used for the model's geometric attention.

---

## Overview of the Graph Transformation

...
---

## Scripts

* **`Load_Data.py`**:
    * Loads the raw CSV.
    * **Crucial:** Handles the specific `MMM-YY` date format to prevent chronological scrambling.

* **`Cyber_Trend_to_Graph.py`**:
    * The main engine. Imports `Load_Data`, applies Double Exponential Smoothing, and generates the `.npz` files.
    * Supports the `--pdformer` flag to generate the computationally expensive matrices (DTW, Clustering).

* **`Cyber_Trend_to_Image.py`**:
    * **Vision Pipeline Only:** Converts the time-series into colorised 2D images for the VisionTS model.

---

## How to Run

**IMPORTANT:** To ensure relative imports work, you must run these scripts as a **module** from the **Project Root** directory (one level up from `Transformer_Pipeline`).

### 1. Graph Pipeline Execution
This generates the full dataset including PDFormer artifacts.

**1.1 Generic Pathway (Graph Base):**
Generates `train.npz`, `val.npz`, `test.npz`, `adj_mx.npy`.

```bash
python -m Transformer_Pipeline.Preprocessing.Cyber_Trend_to_Graph
```
**1.2 Extended Pathway (Graph PDFormer):**
Generates base files plus `dtw_matrix.npy`, `sh_mx.npy`, `pattern_keys.npy`.

```bash
python -m Transformer_Pipeline.Preprocessing.Cyber_Trend_to_Graph --pdformer
```
Outputs (in `Processed_Data/graph/`):

* `train.npz`, `val.npz`, `test.npz` (The 10-input / 36-output tensors)
* `adj_mx.npy` (Adjacency Matrix)
* `dtw_matrix.npy`, `sh_mx.npy`, `pattern_keys.npy` (PDFormer specifics)

### 2. Vision Pipeline Execution
To generate data for the VisionTS experiment:

```bash
python -m Transformer_Pipeline.Preprocessing.Cyber_Trend_to_Image
```