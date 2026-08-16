# Models used in this Project

This directory contains the source code for the transformer models used in our experiment.

## Management via Git Submodules

The models in this directory are managed as **Git Submodules**, not as direct copies of the code.

## Current Submodules

* **`graph-transformer/PDFormer`**:

    * **Model:** PDFormer (Propagation-Delayed Spatio-Temporal Transformer)
    * **Original Repository:** [https://github.com/BUAABIGSCity/PDFormer](https://github.com/BUAABIGSCity/PDFormer)
    * **Paper:** [https://arxiv.org/abs/2301.07945](https://arxiv.org/abs/2301.07945)

* **`visual-transformer/`**:

    * **Model:**
    * **Original Repository:**
    * **Paper:**

## Important: Cloning the Repository

Use the following command to clone the main repository and all its submodules. A standard `git clone` will leave the submodule folders empty.

```bash
git clone --recurse-submodules https://github.com/ehshan/Cyber-trend-forecasting.git
```

If you have already cloned the repository without the submodules, you can initialise them by navigating into the project directory and running:

```bash
git submodule update --init --recursive
```