# Transformer Pipeline

This directory contains all the original source code for the experiments in this project. It includes the scripts for data preprocessing, the definitions of the transformer models, and the main script for training and evaluating the models.

## Directory Structure

```
Transformer_Pipeline/
├── Preprocessing/
│   └── Load_Data.py
│   └── Cyber_Trend_to_Graph.py
└── Models/
    └── example_model.py
```
## Preprocessing Scripts

This directory contains the scripts for loading and transforming the raw cyber threat data into formats suitable for the transformer models.

* **`Load_Data.py`**:
    * Provides a reusable function (`load_cyber_threat_data`) to load the raw `Cyber_Trend_Forecasting_All.csv` file.
    * Handles date parsing and sets the 'Date' column as the DataFrame index.
    * Performs basic validation (checking file existence, sorting data).

* **`Cyber_Trend_to_Graph.py`**:
    * Imports the loading function from `Load_Data.py`.
    *

## How to Run the Pipeline

Follow these steps to preprocess the data and run the training experiments.

1.  **Install Dependencies**
    ```bash
    ...
    ```

2.  **Run Preprocessing**
    ```bash
    ...
    ```

3.  **Run Training**
    ```bash
    ...
    ```

## Dependencies

All Python libraries required to run the code in this pipeline are listed in the `requirements.txt` file at the root of the main project repository. Please ensure you have installed them before running any scripts.
