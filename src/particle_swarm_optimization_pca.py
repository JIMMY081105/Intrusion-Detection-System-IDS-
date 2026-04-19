# File:    particle_swarm_optimization_pca.py
# Purpose: PSO feature selection applied to PCA-reduced CICIDS2017 features.
#          A swarm of particles explores the binary feature space; each particle's position
#          represents a PCA component subset, and fitness is the validation F1-score from RF.
# Input:   data/processed/train_pca_multiclass.csv, data/processed/test_pca_multiclass.csv
# Output:  Per-iteration best/avg validation F1, final test metrics, selected PCA indices.
#          Optional: per-config wandb runs under project "cicids-pso".
#
# PSO overview:
#   1. Initialise swarm with random binary positions and continuous velocities.
#   2. Each particle evaluates its position (feature subset) using validation F1.
#   3. Velocity is updated using inertia, cognitive pull (personal best), and social pull (global best).
#   4. Velocity is passed through a sigmoid to produce update probabilities for binary positions.
#   5. After ITERATIONS, retrain RF on the full training set using the global best position.

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

_train_path = os.path.join(DATA_DIR, "train_pca_multiclass.csv")
_test_path  = os.path.join(DATA_DIR, "test_pca_multiclass.csv")

if not os.path.exists(_train_path) or not os.path.exists(_test_path):
    print(f"[ERROR] Processed PCA dataset not found in: {DATA_DIR}")
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
VAL_SIZE   = 0.20   # fraction of train used as validation during PSO search

PSO_CONFIGS = [
    {"name": "PSO_swarm20_w070_c115_c215", "swarm_size": 20, "w": 0.70, "c1": 1.5, "c2": 1.5},
    {"name": "PSO_swarm20_w050_c115_c215", "swarm_size": 20, "w": 0.50, "c1": 1.5, "c2": 1.5},
    {"name": "PSO_swarm20_w090_c115_c215", "swarm_size": 20, "w": 0.90, "c1": 1.5, "c2": 1.5},
    {"name": "PSO_swarm50_w070_c115_c215", "swarm_size": 50, "w": 0.70, "c1": 1.5, "c2": 1.5},
    {"name": "PSO_swarm50_w050_c115_c215", "swarm_size": 50, "w": 0.50, "c1": 1.5, "c2": 1.5},
    {"name": "PSO_swarm50_w090_c115_c215", "swarm_size": 50, "w": 0.90, "c1": 1.5, "c2": 1.5},
]

# ─── Load Data ────────────────────────────────────────────────────────────────

train_df = pd.read_csv(f"{DATA_DIR}/train_pca_multiclass.csv")
test_df  = pd.read_csv(f"{DATA_DIR}/test_pca_multiclass.csv")

print(f"[INFO] Train samples loaded: {len(train_df)}")
print(f"[INFO] Test samples loaded : {len(test_df)}")

X_train_full = train_df.drop(columns=[TARGET_COLUMN]).values
y_train_full = train_df[TARGET_COLUMN].values

X_test = test_df.drop(columns=[TARGET_COLUMN]).values
y_test = test_df[TARGET_COLUMN].values

# Split training data into inner-train and validation for fitness evaluation.
# The test set is never seen during the PSO search.
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

# ─── PSO Operators ────────────────────────────────────────────────────────────

def sigmoid(x):
    """Convert continuous velocity to a probability for binary position update."""
    return 1.0 / (1.0 + np.exp(-x))


def initialize_swarm(swarm_size, rng):
    """Create initial particle positions (binary) and velocities (continuous)."""
    positions = rng.integers(0, 2, size=(swarm_size, num_features))

    # Ensure no particle starts with all features deselected.
    for i in range(swarm_size):
        if np.sum(positions[i]) == 0:
            random_idx = rng.integers(0, num_features)
            positions[i, random_idx] = 1

    # Velocities are continuous — they steer the probability of flipping each bit.
    velocities = rng.uniform(-1.0, 1.0, size=(swarm_size, num_features))

    return positions, velocities


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


# ─── PSO Search ───────────────────────────────────────────────────────────────

def run_pso_config(config, config_index):
    """Run one full PSO search for a given hyperparameter configuration."""
    swarm_size       = config["swarm_size"]
    inertia_weight   = config["w"]
    cognitive_weight = config["c1"]
    social_weight    = config["c2"]
    config_name      = config["name"]

    # Per-config seed keeps each configuration's search independent and reproducible.
    rng = np.random.default_rng(GLOBAL_SEED + config_index)

    run_name = f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = init_wandb(
        project="cicids-pso",
        name=run_name,
        group="pso_hyperparameter_search",
        tags=["PSO", "validation_search"],
        config={
            "algorithm": "PSO",
            "swarm_size": swarm_size,
            "iterations": ITERATIONS,
            "inertia_weight": inertia_weight,
            "cognitive_weight": cognitive_weight,
            "social_weight": social_weight,
            "features": num_features,
            "dataset": "CICIDS2017 PCA",
            "base_model": "RandomForest",
            "validation_split_inside_train": VAL_SIZE,
            "evaluation_protocol": "PSO search on validation, final test only once",
            "global_seed": GLOBAL_SEED
        }
    )

    positions, velocities = initialize_swarm(swarm_size, rng)

    personal_best_positions = positions.copy()
    personal_best_scores    = np.array([fitness(particle) for particle in positions])

    global_best_idx      = np.argmax(personal_best_scores)
    global_best_position = personal_best_positions[global_best_idx].copy()
    global_best_score    = personal_best_scores[global_best_idx]

    start_total = time.time()

    for iteration in range(ITERATIONS):
        start_iter     = time.time()
        current_scores = []

        for i in range(swarm_size):
            r1 = rng.random(num_features)
            r2 = rng.random(num_features)

            # Standard PSO velocity update: inertia + cognitive + social components.
            velocities[i] = (
                inertia_weight   * velocities[i]
                + cognitive_weight * r1 * (personal_best_positions[i] - positions[i])
                + social_weight    * r2 * (global_best_position - positions[i])
            )

            # Clip velocity to prevent exploding values before sigmoid conversion.
            velocities[i] = np.clip(velocities[i], -6, 6)

            # Binary position update: sigmoid converts velocity to flip probability.
            probabilities = sigmoid(velocities[i])
            random_draw   = rng.random(num_features)
            positions[i]  = (random_draw < probabilities).astype(int)

            # Prevent any particle from ending up with no features selected.
            if np.sum(positions[i]) == 0:
                random_idx = rng.integers(0, num_features)
                positions[i, random_idx] = 1

            current_score = fitness(positions[i])
            current_scores.append(current_score)

            if current_score > personal_best_scores[i]:
                personal_best_scores[i]    = current_score
                personal_best_positions[i] = positions[i].copy()

                if current_score > global_best_score:
                    global_best_score    = current_score
                    global_best_position = positions[i].copy()

        iter_time = time.time() - start_iter

        print(
            f"[{config_name}] Iteration {iteration+1}/{ITERATIONS} "
            f"- Best Val F1: {global_best_score:.6f} - Time: {iter_time:.2f}s"
        )

        run.log({
            "iteration":   iteration + 1,
            "best_val_f1": global_best_score,
            "avg_val_f1":  float(np.mean(current_scores)),
            "iter_time":   iter_time
        })

    total_search_time = time.time() - start_total

    selected_features = np.where(global_best_position == 1)[0]

    # Retrain on the full training set using the best feature subset found by PSO.
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
    print("Best Validation F1           :", global_best_score)

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

        "best_validation_f1":    global_best_score,
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
        "config_name":  config_name,
        "swarm_size":   swarm_size,
        "w":            inertia_weight,
        "c1":           cognitive_weight,
        "c2":           social_weight,
        "best_validation_f1":    global_best_score,
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

print("\n===== START PSO HYPERPARAMETER SEARCH =====")
for idx, config in enumerate(PSO_CONFIGS, start=1):
    print(
        f"\n===== RUNNING CONFIG {idx}/{len(PSO_CONFIGS)} : "
        f"{config['name']} "
        f"(swarm={config['swarm_size']}, w={config['w']}, c1={config['c1']}, c2={config['c2']}) ====="
    )

    result = run_pso_config(config, idx)
    all_results.append(result)

# Select the best configuration based solely on validation performance (not test).
best_result = max(all_results, key=lambda x: x["best_validation_f1"])

print("\n===== ALL CONFIG RESULTS (VALIDATION USED FOR SELECTION) =====")
for result in all_results:
    print(
        f"{result['config_name']} | swarm={result['swarm_size']} | "
        f"w={result['w']} | c1={result['c1']} | c2={result['c2']} | "
        f"val_f1={result['best_validation_f1']:.6f} | "
        f"features={result['num_selected_features']} | "
        f"search_time={result['search_time']:.2f}s"
    )

print("\n===== BEST CONFIG SELECTED =====")
print("Best Config Name            :", best_result["config_name"])
print("Best Swarm Size             :", best_result["swarm_size"])
print("Best Inertia Weight         :", best_result["w"])
print("Best Cognitive Weight       :", best_result["c1"])
print("Best Social Weight          :", best_result["c2"])
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
