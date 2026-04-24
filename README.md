# [Paper Title TBD] Official Implementation

> **Note:** > This repository contains the official code implementation for the paper **"Event-Driven Heterogeneous Hypergraph Learning for Next-Activity Prediction on Object-Centric Event Logs"** *(Title TBD / Under Review)*.

This project proposes an event-driven Heterogeneous Hypergraph Learning approach specifically designed for Object-Centric Event Logs (OCEL) to tackle the Next-Activity Prediction task in process mining.

## 📁 Repository Structure

The core code and datasets are organized as follows:

```text
.
├── data/                       # Directory for storing raw datasets
└── newTry/                     # Core implementation code
    ├── 📊 Data Analysis & Preprocessing
    │   ├── ana.py              # Script for exploring/analyzing raw datasets (helps in writing specific pipelines)
    │   ├── construct_PE.py     # Helper functions for feature construction (called by pipelines)
    │   ├── preprocess.py       # Helper functions for general preprocessing (called by pipelines)
    │   ├── pipeline_OTC.py     # Dedicated preprocessing pipeline for the OTC dataset
    │   ├── pipline_2017.py     # Dedicated preprocessing pipeline for the BPI 2017 dataset
    │   ├── pipline_inter.py    # Dedicated preprocessing pipeline for the Inter dataset
    │   └── pipline_p2p.py      # Dedicated preprocessing pipeline for the P2P dataset
    │
    ├── 🧠 Model Architecture
    │   ├── OCELhg.py           # Custom core data structure: OCEL Heterogeneous Hypergraph
    │   ├── encoder.py          # Encoder module of the model
    │   └── model.py            # Overall model architecture definition
    │
    ├── 🚀 Training & Evaluation
    │   ├── Trainer.py          # Model training script
    │   └── test.py             # Model testing and evaluation script
    │
    └── ⚙️ Configuration & Utilities
        ├── config.py           # Configuration file (hyperparameters, paths, etc.)
        └── utils.py            # General utility functions

Quick Start & Usage Guide
Follow these steps to set up the environment, prepare your data, and run the model.

1. Environment Setup
We recommend using Conda to create a virtual environment:

Bash
conda create -n ocel_hg python=3.8
conda activate ocel_hg
# Install required packages (replace with your actual requirements.txt if available)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install torch_geometric pandas numpy pm4py
2. Data Preparation
Place your raw OCEL datasets (e.g., .jsonocel or .csv files) into the data/ directory. Ensure the filenames match the paths specified in your preprocessing pipelines.

3. Configuration (config.py)
Before running any scripts, open newTry/config.py to configure your experiment settings. Here you can set:

Target Dataset: Specify which dataset you are currently running.

Hyperparameters: Adjust learning_rate, batch_size, hidden_dim, epochs, etc.

Paths: Verify input data paths and output directories for saved models.

4. Data Preprocessing (Crucial First Step)
Because different OCEL datasets vary significantly in structure and attributes, we do not use a universal preprocessing approach. You must run the specific pipeline tailored to your dataset. These scripts will clean the data, extract features, and construct the hypergraph topology.

# Example: If you are using the OTC dataset
python newTry/pipeline_OTC.py

# Or for other datasets:
# python newTry/pipline_2017.py
# python newTry/pipline_inter.py
# python newTry/pipline_p2p.py
Tip for Custom Datasets: If you are introducing a new dataset, run newTry/ana.py first to analyze the raw data structure. This will help you write a custom pipeline script for it.

5. Training
Once the preprocessing is complete and features are saved, you can start training the model. The trainer will automatically read the processed data and the configurations from config.py.

python newTry/Trainer.py
6. Testing & Evaluation
After the model finishes training and the weights are saved, run the test script to evaluate its next-activity prediction performance (e.g., Accuracy, F1-Score) on the test set.

python newTry/test.py
