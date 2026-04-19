# File:    random_forest_raw.py
# Purpose: Baseline Random Forest classifier trained on all raw (non-PCA) CICIDS2017 features.
#          Provides the comparison benchmark for metaheuristic feature selection methods.
# Input:   data/processed/train_multiclass.csv, data/processed/test_multiclass.csv
# Output:  Console — accuracy, precision, recall, F1, balanced accuracy, confusion matrix,
#          train-test gap, per-class FPR/TPR. Optional: wandb experiment tracking.
#
# Random Forest overview:
#   1. Bootstrap sampling: each tree trains on a random sample (with replacement) of the rows.
#   2. Feature subsampling: each split considers sqrt(n_features) candidates for diversity.
#   3. Each of the 100 trees independently assigns a class label to each test sample.
#   4. The majority vote across all trees becomes the final prediction.

import os
import sys

from env_setup import GPU_AVAILABLE, WANDB_AVAILABLE, DATA_DIR, TARGET_COLUMN, init_wandb

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    balanced_accuracy_score,
)
import time

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

# ─── Load Data ────────────────────────────────────────────────────────────────

train_df = pd.read_csv(f"{DATA_DIR}/train_multiclass.csv")
test_df  = pd.read_csv(f"{DATA_DIR}/test_multiclass.csv")

print(f"[INFO] Train samples loaded: {len(train_df)}")
print(f"[INFO] Test samples loaded : {len(test_df)}")

X_train = train_df.drop(columns=[TARGET_COLUMN]).values
y_train = train_df[TARGET_COLUMN].values

X_test = test_df.drop(columns=[TARGET_COLUMN]).values
y_test = test_df[TARGET_COLUMN].values

# ─── Overlap Detection ────────────────────────────────────────────────────────

def exact_overlap_count(X_a, y_a, X_b, y_b):
    """Return the number of rows that appear in both (X_a, y_a) and (X_b, y_b)."""
    a = np.column_stack([X_a, y_a])
    b = np.column_stack([X_b, y_b])
    set_a = {row.tobytes() for row in a}
    set_b = {row.tobytes() for row in b}
    return len(set_a.intersection(set_b))


overlap_count = exact_overlap_count(X_train, y_train, X_test, y_test)
print("\n===== DATA SANITY CHECK =====")
print("Exact Train-Test Overlap Count :", overlap_count)

overlap_removed = 0

if overlap_count != 0:
    print(f"\n[WARNING] Found {overlap_count} overlapping rows between train and test.")

    a     = np.column_stack([X_train, y_train])
    b     = np.column_stack([X_test, y_test])
    set_a = {row.tobytes() for row in a}

    # Keep only test rows that do not appear in the training set.
    mask = np.array([row.tobytes() not in set_a for row in b])

    original_test_size = len(X_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    new_test_size = len(X_test)
    overlap_removed = original_test_size - new_test_size

    print(f"[INFO] Removed {overlap_removed} overlapping rows from TEST set")
    print(f"[INFO] Test set size: {original_test_size} -> {new_test_size}")

# ─── W&B Initialisation ───────────────────────────────────────────────────────

run = init_wandb(
    project="cicids-random-forest",
    name="rf_raw_baseline",
    config={
        "model": "RandomForest",
        "n_estimators": 100,
        "features": X_train.shape[1],
        "dataset": "CICIDS2017 Raw",
        "evaluation_protocol": "train on full train, test once, overlap-safe handling on test if needed",
    }
)

# ─── Train ────────────────────────────────────────────────────────────────────

# n_jobs=-1 uses all CPU cores; cuML intercepts this call automatically if GPU is active.
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

start = time.time()
rf.fit(X_train, y_train)
train_time = time.time() - start

# Predict on train to enable overfitting analysis.
start = time.time()
y_train_pred = rf.predict(X_train)
train_predict_time = time.time() - start

# ─── Evaluate ────────────────────────────────────────────────────────────────

start  = time.time()
y_pred = rf.predict(X_test)
predict_time = time.time() - start

total_time = train_time + predict_time

# ===== TRAIN METRICS =====
train_accuracy          = accuracy_score(y_train, y_train_pred)
train_precision         = precision_score(y_train, y_train_pred, average="macro", zero_division=0)
train_recall            = recall_score(y_train, y_train_pred, average="macro", zero_division=0)
train_f1                = f1_score(y_train, y_train_pred, average="macro", zero_division=0)
train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)

# ===== TEST METRICS =====
accuracy     = accuracy_score(y_test, y_pred)
precision    = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall       = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1           = f1_score(y_test, y_pred, average="macro", zero_division=0)
balanced_acc = balanced_accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

if WANDB_AVAILABLE:
    import wandb
    run.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_test,
            preds=y_pred
        )
    })

# ─── Per-class FPR / TPR ─────────────────────────────────────────────────────

# FPR/TPR are strictly binary metrics; these are one-vs-rest approximations for multiclass.
print("\nPer-class metrics from confusion matrix:")

num_classes = cm.shape[0]

for i in range(num_classes):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0

    print(f"\nClass {i}:")
    print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"FPR={fpr:.6f}, TPR(Recall)={tpr:.6f}")

# ─── Results ─────────────────────────────────────────────────────────────────

n_features = X_train.shape[1]

print("\n===== TRAIN RESULTS =====")
print("Train Accuracy :", train_accuracy)
print("Train Precision:", train_precision)
print("Train Recall   :", train_recall)
print("Train F1 Score :", train_f1)
print("Train Balanced Accuracy :", train_balanced_accuracy)
print("Train Predict Time:", train_predict_time)

print("\n===== TEST RESULTS =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("Balanced Accuracy :", balanced_acc)
print("Train Time :", train_time)
print("Predict Time:", predict_time)
print("Total time:", total_time)
print("Number of Features:", n_features)

print("\n===== OVERLAP HANDLING =====")
print("Overlap Count Detected :", overlap_count)
print("Overlap Removed From Test :", overlap_removed)
print("Final Test Size :", len(y_test))

run.log({
    "overlap_count_detected": overlap_count,
    "overlap_removed_from_test": overlap_removed,
    "final_test_size": len(y_test),

    "train_accuracy": train_accuracy,
    "train_precision": train_precision,
    "train_recall": train_recall,
    "train_f1_score": train_f1,
    "train_balanced_accuracy": train_balanced_accuracy,
    "train_predict_time": train_predict_time,

    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "balanced_accuracy": balanced_acc,
    "train_time": train_time,
    "predict_time": predict_time,
    "total_time": total_time,
    "num_features": n_features
})

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n===== TRAIN-TEST GAP =====")
print("Accuracy Gap :", train_accuracy - accuracy)
print("Precision Gap:", train_precision - precision)
print("Recall Gap   :", train_recall - recall)
print("F1 Gap       :", train_f1 - f1)
print("Balanced Acc Gap :", train_balanced_accuracy - balanced_acc)

run.finish()
