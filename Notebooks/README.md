# Notebook Vignettes

Directory contains Jupyter notebooks used for verifying pipeline logic, visualising data transformations, and interactively debugging the model architectures.

## Notebooks

### 1. `Graph_Pipeline_Verification.ipynb`
**Purpose:** The primary test for the Graph (PDFormer) pipeline.
**Key Functions:**
* **Data Verification:** Visualises the `MMM-YY` date sorting and the effect of Double Exponential Smoothing (DES).
* **Tensor Shapes:** Confirms that the Preprocessing script generates strict 10-month input and 36-month target tensors.
* **Training Loop:** Runs a short, interactive training session to verify that the RSE/RAE metrics are calculating correctly and decreasing.

### 2. `Transformer_Pipeline_Vignette.ipynb`
**Purpose:** Pipeline dashboard and demonstration tool for the project. Executes the end-to-end workflow (Train → Eval → Viz) and is designed to run on both Google Colab and the natogpu server.
**Key Functions:**
* **Environment Switching:** Automatically detects the runtime (Colab vs. Local) and handles drive mounting, pathing, and dependencies accordingly.
* **Pipeline Manager:** Sequentially executes `Train_Graph.py`, `Evaluate_Graph.py`, and `Visualise_Results.py` using isolated memory states
* **Interactive Demo:** Displays the final "Gap Analysis" and "Forecast" plots inline for stakeholder review.

## Setup & Usage

* **Dependencies:** Ensure the full project requirements are installed, specifically `matplotlib`, `seaborn`, and `tqdm` for the visualisations.

    ```bash
    pip install -r ../requirements.txt
    ```
* **Pathing:** These notebooks include a `sys.path` setup block at the top to allow importing modules from the sibling directory (`../Transformer_Pipeline`). Please do not remove this block.