# File:    simulated_annealing_raw.py
# Purpose: Simulated Annealing feature selection applied to raw (non-PCA) CICIDS2017 features.
#          Starts from a random feature subset and explores neighbours with controlled
#          acceptance of worse solutions (governed by temperature) to escape local optima.
# Input:   data/processed/train_multiclass.csv, data/processed/test_multiclass.csv
# Output:  Per-iteration best/current validation F1, temperature trace, final test metrics.
#          Optional: per-config wandb runs under project "cicids-sa-raw".
#
# SA overview:
#   1. Initialise a random binary feature mask as the current solution.
#   2. Each iteration: generate a neighbour by flipping one or more feature bits.
#   3. Accept the neighbour if it is better; accept it with probability exp(delta/T) if worse.
#   4. Cool the temperature by the cooling rate after each iteration.
#   5. After ITERATIONS, retrain RF on the full training set using the best solution found.

import os
import sys

from env_setup import GPU_AVAILABLE, WANDB_AVAILABLE, DATA_DIR, TARGET_COLUMN, init_wandb

import pandas as pd
import numpy as np
import random
import time
import warnings
from datetime import datetime

from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ─── Dataset Check ────────────────────────────────────────────────────────────

_train_path = os.path.join(DATA_DIR, "train_multiclass.csv")
_test_path  = os.path.join(DATA_DIR, "test_multiclass.csv")

if not os.path.exists(_train_path) or not os.path.exists(_test_path):
    print(f"[ERROR] Processed dataset not found in: {DATA_DIR}")
    print("[ERROR] Run src/dataset_prepare.py first to generate the datasets.")
    sys.exit(1)

print(f"[INFO] Dataset path : {DATA_DIR}")
print(f"[INFO] Loading train: {_train_path}")
print(f"[INFO] Loading test : {_test_path}")

# ─── Reproducibility ─────────────────────────────────────────────────────────

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ─── Hyperparameters ─────────────────────────────────────────────────────────

ITERATIONS = 15
VAL_SIZE   = 0.20   # fraction of train used as validation during SA search

SA_CONFIGS = [
    {"name": "SA_RAW_temp20_cool095_flip1", "initial_temp": 2.0, "cooling_rate": 0.95, "flip_bits": 1},
    {"name": "SA_RAW_temp20_cool090_flip1", "initial_temp": 2.0, "cooling_rate": 0.90, "flip_bits": 1},
    {"name": "SA_RAW_temp20_cool095_flip2", "initial_temp": 2.0, "cooling_rate": 0.95, "flip_bits": 2},
    {"name": "SA_RAW_temp50_cool095_flip1", "initial_temp": 5.0, "cooling_rate": 0.95, "flip_bits": 1},
    {"name": "SA_RAW_temp50_cool090_flip1", "initial_temp": 5.0, "cooling_rate": 0.90, "flip_bits": 1},
    {"name": "SA_RAW_temp50_cool095_flip2", "initial_temp": 5.0, "cooling_rate": 0.95, "flip_bits": 2},
]

# ─── Load Data ────────────────────────────────────────────────────────────────

train_df = pd.read_csv(f"{DATA_DIR}/train_multiclass.csv")
test_df  = pd.read_csv(f"{DATA_DIR}/test_multiclass.csv")

print(f"[INFO] Train samples loaded: {len(train_df)}")
print(f"[INFO] Test samples loaded : {len(test_df)}")

X_train_full = train_df.drop(columns=[TARGET_COLUMN]).values
y_train_full = train_df[TARGET_COLUMN].values

X_test = test_df.drop(columns=[TARGET_COLUMN]).values
y_test = test_df[TARGET_COLUMN].values

# Split training data into inner-train and validation for fitness evaluation.
# The test set is never seen during the SA search.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=VAL_SIZE,
    random_state=GLOBAL_SEED,
    stratify=y_train_full
)

num_features = X_train_full.shape[1]

# ─── Overlap Detection ────────────────────────────────────────────────────────

def exact_overlap_count(X_a, y_a, X_b, y_b):
    """Return the number of rows that appear in both (X_a, y_a) and (X_b, y_b)."""
    a = np.column_stack([X_a, y_a])
    b = np.column_stack([X_b, y_b])
    set_a = {row.tobytes() for row in a}
    set_b = {row.tobytes() for row in b}
    return len(set_a.intersection(set_b))


overlap_count = exact_overlap_count(X_train_full, y_train_full, X_test, y_test)
print("\n===== DATA SANITY CHECK =====")
print("Exact Train-Test Overlap Count :", overlap_count)

overlap_removed = 0

if overlap_count != 0:
    print(f"\n[WARNING] Found {overlap_count} overlapping rows between train and test.")

    a     = np.column_stack([X_train_full, y_train_full])
    b     = np.column_stack([X_test, y_test])
    set_a = {row.tobytes() for row in a}

    mask = np.array([row.tobytes() not in set_a for row in b])

    original_test_size = len(X_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    new_test_size = len(X_test)
    overlap_removed = original_test_size - new_test_size

    print(f"[INFO] Removed {overlap_removed} overlapping rows from TEST set")
    print(f"[INFO] Test set size: {original_test_size} -> {new_test_size}")

# ─── SA Operators ─────────────────────────────────────────────────────────────

def fitness(individual):
    """Evaluate a feature mask by training RF on inner-train and scoring on validation."""
    if np.sum(individual) == 0:
        return 0

    selected_features = np.where(individual == 1)[0]

    X_tr = X_train[:, selected_features]
    X_va = X_val[:, selected_features]

    # RF is used consistently with the baseline for a fair comparison.
    # cuml.accel redirects this to GPU automatically if available.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_va)

    return f1_score(y_val, y_pred, average="macro", zero_division=0)


def initialize_solution(rng):
    """Create a random binary feature mask as the starting solution."""
    solution = rng.integers(0, 2, size=num_features)

    # Ensure the starting solution is not empty.
    if np.sum(solution) == 0:
        random_idx = rng.integers(0, num_features)
        solution[random_idx] = 1

    return solution


def generate_neighbour(solution, flip_bits, rng):
    """Produce a neighbour by randomly flipping flip_bits feature bits."""
    neighbour = solution.copy()

    flip_indices = rng.choice(num_features, size=flip_bits, replace=False)
    for idx in flip_indices:
        neighbour[idx] = 1 - neighbour[idx]

    # Prevent the neighbour from becoming an empty feature mask.
    if np.sum(neighbour) == 0:
        random_idx = rng.integers(0, num_features)
        neighbour[random_idx] = 1

    return neighbour


# ─── SA Search ────────────────────────────────────────────────────────────────

def run_sa_config(config, config_index):
    """Run one full SA search for a given hyperparameter configuration."""
    initial_temp  = config["initial_temp"]
    cooling_rate  = config["cooling_rate"]
    flip_bits     = config["flip_bits"]
    config_name   = config["name"]

    # Per-config seed keeps each configuration's search independent and reproducible.
    rng = np.random.default_rng(GLOBAL_SEED + config_index)

    run_name = f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = init_wandb(
        project="cicids-sa-raw",
        name=run_name,
        group="sa_raw_hyperparameter_search",
        tags=["SA", "validation_search", "raw_data"],
        config={
            "algorithm": "SA",
            "iterations": ITERATIONS,
            "initial_temp": initial_temp,
            "cooling_rate": cooling_rate,
            "flip_bits": flip_bits,
            "features": num_features,
            "dataset": "CICIDS2017 Raw",
            "base_model": "RandomForest",
            "validation_split_inside_train": VAL_SIZE,
            "evaluation_protocol": "SA search on validation, final test only once",
            "global_seed": GLOBAL_SEED
        }
    )

    current_solution = initialize_solution(rng)
    current_score    = fitness(current_solution)

    best_solution  = current_solution.copy()
    best_score     = current_score
    best_iteration = 0

    temperature = initial_temp

    start_total = time.time()

    for iteration in range(ITERATIONS):
        start_iter = time.time()

        neighbour       = generate_neighbour(current_solution, flip_bits, rng)
        neighbour_score = fitness(neighbour)

        delta = neighbour_score - current_score

        # Always accept an improvement; accept a worse solution with probability exp(delta/T).
        if delta >= 0:
            current_solution = neighbour.copy()
            current_score    = neighbour_score
        else:
            acceptance_probability = np.exp(delta / temperature) if temperature > 1e-12 else 0
            if rng.random() < acceptance_probability:
                current_solution = neighbour.copy()
                current_score    = neighbour_score

        if current_score > best_score:
            best_score     = current_score
            best_solution  = current_solution.copy()
            best_iteration = iteration + 1

        # Reduce temperature to gradually lower the probability of accepting worse solutions.
        temperature = temperature * cooling_rate

        iter_time = time.time() - start_iter

        print(
            f"[{config_name}] Iteration {iteration+1}/{ITERATIONS} "
            f"- Best Val F1: {best_score:.6f} - Time: {iter_time:.2f}s"
        )

        run.log({
            "iteration":       iteration + 1,
            "best_val_f1":     best_score,
            "current_val_f1":  current_score,
            "temperature":     temperature,
            "iter_time":       iter_time
        })

    total_search_time = time.time() - start_total

    selected_features = np.where(best_solution == 1)[0]

    # Retrain on the full training set using the best feature subset found by SA.
    X_tr_final = X_train_full[:, selected_features]
    X_te_final = X_test[:, selected_features]

    start_train = time.time()
    final_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_tr_final, y_train_full)
    train_time = time.time() - start_train

    start_test = time.time()
    y_pred = final_model.predict(X_te_final)
    test_time = time.time() - start_test

    y_train_pred = final_model.predict(X_tr_final)

    # ===== TEST METRICS =====
    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec     = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1      = f1_score(y_test, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    # ===== TRAIN METRICS (for overfitting analysis) =====
    train_acc     = accuracy_score(y_train_full, y_train_pred)
    train_prec    = precision_score(y_train_full, y_train_pred, average="macro", zero_division=0)
    train_rec     = recall_score(y_train_full, y_train_pred, average="macro", zero_division=0)
    train_f1      = f1_score(y_train_full, y_train_pred, average="macro", zero_division=0)
    train_bal_acc = balanced_accuracy_score(y_train_full, y_train_pred)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== [{config_name}] CLASSIFICATION REPORT =====\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    print(f"\n===== [{config_name}] FINAL RESULTS =====")
    print("Selected Features            :", selected_features)
    print("Number of Selected Features  :", len(selected_features))
    print("Best Validation F1           :", best_score)

    print("\n----- TRAIN METRICS -----")
    print("Train Accuracy          :", train_acc)
    print("Train Precision (macro) :", train_prec)
    print("Train Recall (macro)    :", train_rec)
    print("Train F1 Score (macro)  :", train_f1)
    print("Train Balanced Accuracy :", train_bal_acc)

    print("\n----- TEST METRICS -----")
    print("Accuracy          :", acc)
    print("Precision (macro) :", prec)
    print("Recall (macro)    :", rec)
    print("F1 Score (macro)  :", f1)
    print("Balanced Accuracy :", bal_acc)
    print("Train Time        :", train_time)
    print("Test Time         :", test_time)
    print("Total Search Time :", total_search_time)

    print("\n----- TRAIN-TEST GAP -----")
    print("Accuracy Gap          :", train_acc - acc)
    print("Precision Gap (macro) :", train_prec - prec)
    print("Recall Gap (macro)    :", train_rec - rec)
    print("F1 Gap (macro)        :", train_f1 - f1)
    print("Balanced Accuracy Gap :", train_bal_acc - bal_acc)

    run.log({
        "overlap_count_detected": overlap_count,
        "overlap_removed_from_test": overlap_removed,

        "best_validation_f1":    best_score,
        "best_iteration":        best_iteration,
        "num_selected_features": len(selected_features),

        "train_accuracy":          train_acc,
        "train_precision":         train_prec,
        "train_recall":            train_rec,
        "train_f1":                train_f1,
        "train_balanced_accuracy": train_bal_acc,

        "final_accuracy":          acc,
        "final_precision":         prec,
        "final_recall":            rec,
        "final_f1":                f1,
        "final_balanced_accuracy": bal_acc,

        "train_time":        train_time,
        "test_time":         test_time,
        "total_search_time": total_search_time
    })

    if WANDB_AVAILABLE:
        import wandb
        run.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_test,
                preds=y_pred
            )
        })

    run.finish()

    return {
        "config_name":    config_name,
        "initial_temp":   initial_temp,
        "cooling_rate":   cooling_rate,
        "flip_bits":      flip_bits,
        "best_validation_f1": best_score,
        "best_iteration": best_iteration,
        "selected_features":     selected_features,
        "num_selected_features": len(selected_features),
        "search_time":           total_search_time,

        "train_accuracy":          train_acc,
        "train_precision":         train_prec,
        "train_recall":            train_rec,
        "train_f1":                train_f1,
        "train_balanced_accuracy": train_bal_acc,

        "final_accuracy":          acc,
        "final_precision":         prec,
        "final_recall":            rec,
        "final_f1":                f1,
        "final_balanced_accuracy": bal_acc,

        "train_time":       train_time,
        "test_time":        test_time,
        "confusion_matrix": cm
    }


# ─── Run All Configs ──────────────────────────────────────────────────────────

all_results = []

print("\n===== START SA HYPERPARAMETER SEARCH =====")
for idx, config in enumerate(SA_CONFIGS, start=1):
    print(
        f"\n===== RUNNING CONFIG {idx}/{len(SA_CONFIGS)} : "
        f"{config['name']} "
        f"(temp={config['initial_temp']}, cool={config['cooling_rate']}, flip={config['flip_bits']}) ====="
    )

    result = run_sa_config(config, idx)
    all_results.append(result)

# Select the best configuration based solely on validation performance (not test).
best_result = max(all_results, key=lambda x: x["best_validation_f1"])

print("\n===== ALL CONFIG RESULTS (VALIDATION USED FOR SELECTION) =====")
for result in all_results:
    print(
        f"{result['config_name']} | temp={result['initial_temp']} | "
        f"cool={result['cooling_rate']} | flip={result['flip_bits']} | "
        f"val_f1={result['best_validation_f1']:.6f} | "
        f"features={result['num_selected_features']} | "
        f"search_time={result['search_time']:.2f}s"
    )

print("\n===== BEST CONFIG SELECTED =====")
print("Best Config Name            :", best_result["config_name"])
print("Best Initial Temperature    :", best_result["initial_temp"])
print("Best Cooling Rate           :", best_result["cooling_rate"])
print("Best Flip Bits              :", best_result["flip_bits"])
print("Best Validation F1          :", best_result["best_validation_f1"])
print("Selected Features           :", best_result["selected_features"])
print("Number of Selected Features :", best_result["num_selected_features"])

print("\n===== FINAL BEST CONFIG TEST RESULTS =====")
print("Accuracy          :", best_result["final_accuracy"])
print("Precision (macro) :", best_result["final_precision"])
print("Recall (macro)    :", best_result["final_recall"])
print("F1 Score (macro)  :", best_result["final_f1"])
print("Balanced Accuracy :", best_result["final_balanced_accuracy"])
print("Train Time        :", best_result["train_time"])
print("Test Time         :", best_result["test_time"])
print("Search Time       :", best_result["search_time"])
