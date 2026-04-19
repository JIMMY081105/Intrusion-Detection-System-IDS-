# File:    genetic_algorithm_raw.py
# Purpose: Genetic Algorithm feature selection applied to raw (non-PCA) CICIDS2017 features.
#          Evolves binary feature masks to maximise validation F1-score, then evaluates the
#          best-found subset on the held-out test set using a Random Forest classifier.
# Input:   data/processed/train_multiclass.csv, data/processed/test_multiclass.csv
# Output:  Per-generation validation F1, final test metrics, selected feature indices.
#          Optional: per-config wandb runs under project "cicids-ga-raw".
#
# GA overview:
#   1. Initialise a population of random binary feature masks.
#   2. Evaluate each mask by training RF on inner-train and scoring on a validation split.
#   3. Select parents via tournament selection; produce children via single-point crossover.
#   4. Mutate children by randomly flipping individual feature bits.
#   5. Repeat for GENERATIONS; retain the best mask via elitism.
#   6. Retrain on the full training set using the best mask; evaluate once on the test set.

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

GENERATIONS    = 15
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 3
VAL_SIZE       = 0.20   # fraction of train used as validation during GA search

GA_CONFIGS = [
    {"name": "GA_RAW_pop20_mut005", "population_size": 20, "mutation_rate": 0.05},
    {"name": "GA_RAW_pop20_mut010", "population_size": 20, "mutation_rate": 0.10},
    {"name": "GA_RAW_pop20_mut020", "population_size": 20, "mutation_rate": 0.20},
    {"name": "GA_RAW_pop50_mut005", "population_size": 50, "mutation_rate": 0.05},
    {"name": "GA_RAW_pop50_mut010", "population_size": 50, "mutation_rate": 0.10},
    {"name": "GA_RAW_pop50_mut020", "population_size": 50, "mutation_rate": 0.20},
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
# The test set is never seen during the GA search.
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

# ─── GA Operators ─────────────────────────────────────────────────────────────

def initialize_population(pop_size, rng):
    """Create an initial population of random binary feature masks."""
    population = rng.integers(0, 2, (pop_size, num_features))

    # Ensure no individual starts with all features deselected.
    for i in range(pop_size):
        if np.sum(population[i]) == 0:
            random_idx = rng.integers(0, num_features)
            population[i, random_idx] = 1

    return population


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


def tournament_selection(population, fitness_scores, pop_size, rng):
    """Select individuals via tournament: pick TOURNAMENT_SIZE candidates, keep the best."""
    selected = []
    for _ in range(pop_size):
        candidates = rng.choice(pop_size, size=TOURNAMENT_SIZE, replace=False)
        best = max(candidates, key=lambda idx: fitness_scores[idx])
        selected.append(population[best].copy())
    return np.array(selected)


def crossover(parent1, parent2, pop_rng):
    """Single-point crossover: swap gene segments after a random split point."""
    if pop_rng.random() < CROSSOVER_RATE:
        point  = pop_rng.integers(1, num_features)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()


def mutate(individual, mutation_rate, pop_rng):
    """Flip each feature bit independently with probability mutation_rate."""
    for i in range(num_features):
        if pop_rng.random() < mutation_rate:
            individual[i] = 1 - individual[i]

    # Prevent mutation from producing an empty (all-zero) feature mask.
    if np.sum(individual) == 0:
        random_idx = pop_rng.integers(0, num_features)
        individual[random_idx] = 1

    return individual


# ─── GA Search ────────────────────────────────────────────────────────────────

def run_ga_config(config, config_index):
    """Run one full GA search for a given hyperparameter configuration."""
    pop_size      = config["population_size"]
    mutation_rate = config["mutation_rate"]
    config_name   = config["name"]

    # Per-config seed keeps each configuration's search independent and reproducible.
    rng = np.random.default_rng(GLOBAL_SEED + config_index)

    run_name = f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = init_wandb(
        project="cicids-ga-raw",
        name=run_name,
        group="ga_raw_hyperparameter_search",
        tags=["GA", "validation_search", "raw_data"],
        config={
            "algorithm": "GA",
            "population_size": pop_size,
            "mutation_rate": mutation_rate,
            "generations": GENERATIONS,
            "crossover_rate": CROSSOVER_RATE,
            "tournament_size": TOURNAMENT_SIZE,
            "features": num_features,
            "dataset": "CICIDS2017 Raw",
            "base_model": "RandomForest",
            "validation_split_inside_train": VAL_SIZE,
            "evaluation_protocol": "GA search on validation, final test only once",
            "global_seed": GLOBAL_SEED
        }
    )

    population = initialize_population(pop_size, rng)

    best_individual = None
    best_fitness    = -1
    best_generation = -1

    config_start_total = time.time()

    for gen in range(GENERATIONS):
        start_gen = time.time()

        fitness_scores = np.array([fitness(ind) for ind in population])

        gen_best_idx     = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]

        if gen_best_fitness > best_fitness:
            best_fitness    = gen_best_fitness
            best_individual = population[gen_best_idx].copy()
            best_generation = gen + 1

        gen_time = time.time() - start_gen
        print(
            f"[{config_name}] Generation {gen+1}/{GENERATIONS} "
            f"- Best Val F1: {gen_best_fitness:.6f} - Time: {gen_time:.2f}s"
        )

        run.log({
            "generation":   gen + 1,
            "best_val_f1":  gen_best_fitness,
            "avg_val_f1":   float(np.mean(fitness_scores)),
            "gen_time":     gen_time
        })

        selected_population = tournament_selection(population, fitness_scores, pop_size, rng)
        next_population = []

        # Elitism: carry the best individual forward unchanged to avoid losing progress.
        next_population.append(best_individual.copy())

        while len(next_population) < pop_size:
            parent1 = selected_population[rng.integers(0, pop_size)]
            parent2 = selected_population[rng.integers(0, pop_size)]

            child1, child2 = crossover(parent1, parent2, rng)

            child1 = mutate(child1, mutation_rate, rng)
            child2 = mutate(child2, mutation_rate, rng)

            next_population.append(child1)
            if len(next_population) < pop_size:
                next_population.append(child2)

        population = np.array(next_population)

    total_search_time = time.time() - config_start_total

    selected_features = np.where(best_individual == 1)[0]

    # Retrain on the full training set using the best feature subset found by GA.
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
    print("Best Validation F1           :", best_fitness)

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

        "best_validation_f1":    best_fitness,
        "best_generation":       best_generation,
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

        "train_time":       train_time,
        "test_time":        test_time,
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
        "config_name":       config_name,
        "population_size":   pop_size,
        "mutation_rate":     mutation_rate,
        "best_validation_f1": best_fitness,
        "best_generation":   best_generation,
        "selected_features": selected_features,
        "num_selected_features": len(selected_features),
        "search_time":       total_search_time,

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

        "train_time":      train_time,
        "test_time":       test_time,
        "confusion_matrix": cm
    }


# ─── Run All Configs ──────────────────────────────────────────────────────────

all_results = []

print("\n===== START GA HYPERPARAMETER SEARCH =====")
for idx, config in enumerate(GA_CONFIGS, start=1):
    print(
        f"\n===== RUNNING CONFIG {idx}/{len(GA_CONFIGS)} : "
        f"{config['name']} "
        f"(pop={config['population_size']}, mut={config['mutation_rate']}) ====="
    )

    result = run_ga_config(config, idx)
    all_results.append(result)

# Select the best configuration based solely on validation performance (not test).
best_result = max(all_results, key=lambda x: x["best_validation_f1"])

print("\n===== ALL CONFIG RESULTS (VALIDATION USED FOR SELECTION) =====")
for result in all_results:
    print(
        f"{result['config_name']} | pop={result['population_size']} | "
        f"mut={result['mutation_rate']} | "
        f"val_f1={result['best_validation_f1']:.6f} | "
        f"features={result['num_selected_features']} | "
        f"search_time={result['search_time']:.2f}s"
    )

print("\n===== BEST CONFIG SELECTED =====")
print("Best Config Name            :", best_result["config_name"])
print("Best Population Size        :", best_result["population_size"])
print("Best Mutation Rate          :", best_result["mutation_rate"])
print("Best Validation F1          :", best_result["best_validation_f1"])
print("Best Generation             :", best_result["best_generation"])
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
