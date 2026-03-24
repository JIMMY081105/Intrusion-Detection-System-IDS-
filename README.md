# Intrusion Detection System Using Metaheuristic Feature Selection

A comparative study of metaheuristic-based feature selection methods for network intrusion detection on the CICIDS2017 dataset. The project evaluates Genetic Algorithm (GA), Particle Swarm Optimization (PSO), and Simulated Annealing (SA) against a baseline Random Forest classifier, all built on PCA-reduced features.

## Project Structure

```
machine_learning/
├── src/
│   ├── env_setup.py                  # GPU and W&B environment configuration
│   ├── dataset_prepare.py            # Data preprocessing and PCA pipeline
│   ├── random_forest.py              # Baseline Random Forest classifier
│   ├── genetic_algorithm.py          # GA-based feature selection
│   ├── particle_swarm_optimization.py# PSO-based feature selection
│   └── simulated_annealing.py        # SA-based feature selection
├── data/
│   ├── raw/                          # Raw CICIDS2017 CSV files (not tracked)
│   └── processed/                    # Generated datasets (not tracked)
├── terminal responses/               # Logged terminal output from each run
└── README.md
```

## Dataset

This project uses the **CICIDS2017** (Canadian Institute for Cybersecurity Intrusion Detection System) dataset.

1. Download the dataset from: https://www.unb.ca/cic/datasets/ids-2017.html
2. Place all raw CSV files into the `data/raw/` directory.

The preprocessing pipeline will handle merging, cleaning, label encoding, scaling, and PCA reduction automatically.

## Environment Setup

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended for cuML acceleration)

### Install Dependencies

```bash
pip install pandas numpy scikit-learn
pip install wandb
pip install cuml-cu12
```

> `cuml-cu12` requires a compatible CUDA toolkit. Adjust the package suffix to match your CUDA version (e.g., `cuml-cu11` for CUDA 11). See the [RAPIDS installation guide](https://docs.rapids.ai/install) for details.

### GPU Activation

To run scripts with GPU acceleration via cuML, set the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
CUDA_VISIBLE_DEVICES=0 python <script_name>.py
```

The `env_setup.py` module will automatically detect and enable GPU support if cuML is available. If no GPU is found, it falls back to CPU-based scikit-learn.

### Weights & Biases (W&B)

Experiment tracking uses [Weights & Biases](https://wandb.ai). Log in before running:

```bash
wandb login
```

If W&B is not installed or not logged in, the scripts will still run — tracking is skipped gracefully.

## Running the Scripts

Run all scripts from the `src/` directory.

### 1. Prepare the Dataset

```bash
python dataset_prepare.py
```

This reads the raw CSVs from `data/raw/`, performs cleaning, stratified train-test splitting (80/20), standard scaling, and PCA (retaining 80% variance). Outputs are saved to `data/processed/`.

### 2. Run the Baseline

```bash
CUDA_VISIBLE_DEVICES=0 python random_forest.py
```

Trains a Random Forest (100 trees) on the PCA-reduced training set and evaluates on the test set.

### 3. Run Metaheuristic Feature Selection

Each script runs a hyperparameter grid of 6 configurations, selects the best by validation F1-score, and retrains a final model on the full training set.

```bash
CUDA_VISIBLE_DEVICES=0 python genetic_algorithm.py
CUDA_VISIBLE_DEVICES=0 python particle_swarm_optimization.py
CUDA_VISIBLE_DEVICES=0 python simulated_annealing.py
```

## Method Overview

| Method | Approach | Key Parameters |
|--------|----------|----------------|
| **Random Forest** | Baseline classifier on all PCA features | 100 trees |
| **Genetic Algorithm** | Binary feature masks evolved via tournament selection, crossover, and mutation | Population: 20/50, Mutation rate: 0.05/0.10/0.20 |
| **Particle Swarm Optimization** | Swarm of particles exploring feature space with velocity updates | Swarm: 20/50, Inertia: 0.50/0.70/0.90 |
| **Simulated Annealing** | Single-solution search with probabilistic acceptance of worse neighbours | Temperature: 2.0/5.0, Cooling: 0.90/0.95 |

All metaheuristics use a Random Forest (100 trees) as the underlying classifier for fair comparison. Feature selection fitness is evaluated on a validation split (20% of training data) to prevent test set leakage. Each method runs for 15 iterations/generations.

## Evaluation Metrics

- Accuracy, Precision (macro), Recall (macro), F1-score (macro)
- Balanced Accuracy
- Per-class True Positive Rate and False Positive Rate
- Confusion Matrix
- Train-test gap analysis for overfitting detection
