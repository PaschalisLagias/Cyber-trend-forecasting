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
**Purpose:** Pipeline dashboard and demonstration tool for the project. Executes the end-to-end workflow (Data Prep → Train → Eval → Vis) and is designed to run on both Google Colab and the natogpu server.
**Key Functions:**
* **Environment Switching:** Automatically detects the runtime (Colab vs. Local) and handles drive mounting, pathing, and dependencies accordingly.
* **Data Regeneration:** capability to re-run the graph data generation process to ensure input tensors are built from the latest source CSVs.
* **Pipeline Manager:** Sequentially executes the full model pipeline (`Train_Graph.py`, `Evaluate_Graph.py`) using isolated memory states.
* **Interactive Visualisations:** A modular, multi-cell dashboard that presents results in four distinct layers:
    * **Broad Analysis:** Global aggregate plots for "Forecast Accuracy" and "Gap Analysis" (Threat vs. Solution).
    * **Validation Figures:** Detailed "Forecast vs Actual" plots for key nodes (Fig 3) and a complete grid view of all validation nodes.
    * **Risk Tables:** Tables identifying the top "Widening" (Risks) and "Narrowing" (Successes) gaps.
    * **Continuous Trends:** Long-term historical and forecast evolution (2011–2025) for major threat vectors (Fig 4).

## Setup & Usage

* **Dependencies:** Ensure the full project requirements are installed, specifically `matplotlib`, `seaborn`, and `tqdm` for the visualisations.

    ```bash
    pip install -r ../requirements.txt
    ```
* **Pathing:** These notebooks include a `sys.path` setup block at the top to allow importing modules from the sibling directory (`../Transformer_Pipeline`). Please do not remove this block.