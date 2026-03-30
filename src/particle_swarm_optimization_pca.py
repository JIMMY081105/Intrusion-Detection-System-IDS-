#the core of particle swarm optimization is to treat each particle as one candidate solution
#each particle updates its movement by following personal best and global best
#through iterations position update and velocity update until a better subset is found
#(high score version: one wandb run per config, validation for search, test only once, safe overlap handling)

#allow gpu run - graceful fallback if cuML not available
from env_setup import GPU_AVAILABLE, WANDB_AVAILABLE, DATA_DIR, TARGET_COLUMN, init_wandb

#get pandas to read the csv
import pandas as pd

#get numpy for array and matrix operations
import numpy as np

#get random for reproducibility
import random

#get time to record the running time
import time

#(for unique wandb run names)
from datetime import datetime

#remove warning here
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

#import sklearn for generating confusion matrix
from sklearn.metrics import confusion_matrix

#random forest act as the base model for evaluating feature subsets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
)

#(split train into train_inner and val_inner so PSO search never touches the test set)
from sklearn.model_selection import train_test_split


#(global seed for reproducibility)
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

#define number of iterations
ITERATIONS = 15

#(validation split ratio used only inside training data for PSO fitness)
VAL_SIZE = 0.20

#(hyperparameter study for PSO)
PSO_CONFIGS = [
    {
        "name": "PSO_swarm20_w070_c115_c215",
        "swarm_size": 20,
        "w": 0.70,
        "c1": 1.5,
        "c2": 1.5
    },
    {
        "name": "PSO_swarm20_w050_c115_c215",
        "swarm_size": 20,
        "w": 0.50,
        "c1": 1.5,
        "c2": 1.5
    },
    {
        "name": "PSO_swarm20_w090_c115_c215",
        "swarm_size": 20,
        "w": 0.90,
        "c1": 1.5,
        "c2": 1.5
    },
    {
        "name": "PSO_swarm50_w070_c115_c215",
        "swarm_size": 50,
        "w": 0.70,
        "c1": 1.5,
        "c2": 1.5
    },
    {
        "name": "PSO_swarm50_w050_c115_c215",
        "swarm_size": 50,
        "w": 0.50,
        "c1": 1.5,
        "c2": 1.5
    },
    {
        "name": "PSO_swarm50_w090_c115_c215",
        "swarm_size": 50,
        "w": 0.90,
        "c1": 1.5,
        "c2": 1.5
    }
]

#get the train pca dataset
train_df = pd.read_csv(f"{DATA_DIR}/train_pca_multiclass.csv")
#get the test pca dataset
test_df = pd.read_csv(f"{DATA_DIR}/test_pca_multiclass.csv")

#remove the label column for the machine learning
X_train_full = train_df.drop(columns=[TARGET_COLUMN]).values
#get the values of target using numpy
y_train_full = train_df[TARGET_COLUMN].values

#remove the label column for testing the model
X_test = test_df.drop(columns=[TARGET_COLUMN]).values
#same as train data here
y_test = test_df[TARGET_COLUMN].values

#(split the original training set into inner-train and validation for PSO search only)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=VAL_SIZE,
    random_state=GLOBAL_SEED,
    stratify=y_train_full
)

#define number of features as the num of column in train dataset
#where the target as already being removed
num_features = X_train_full.shape[1]


#(sanity check to avoid hidden train-test overlap problems)
def exact_overlap_count(X_a, y_a, X_b, y_b):
    #combine features and labels together so overlap check is strict
    a = np.column_stack([X_a, y_a])
    b = np.column_stack([X_b, y_b])

    #convert each row into bytes for exact set comparison
    set_a = {row.tobytes() for row in a}
    set_b = {row.tobytes() for row in b}

    return len(set_a.intersection(set_b))


#(check overlap once before running PSO so no accidental leakage slips in)
overlap_count = exact_overlap_count(X_train_full, y_train_full, X_test, y_test)
print("\n===== DATA SANITY CHECK =====")
print("Exact Train-Test Overlap Count :", overlap_count)

overlap_removed = 0

if overlap_count != 0:
    print(f"\n[WARNING] Found {overlap_count} overlapping rows between train and test.")

    #combine features and labels together so overlap check is strict
    a = np.column_stack([X_train_full, y_train_full])
    b = np.column_stack([X_test, y_test])

    #convert each row into bytes for exact set comparison
    set_a = {row.tobytes() for row in a}

    #create mask to keep only non-overlapping rows in test set
    mask = np.array([row.tobytes() not in set_a for row in b])

    original_test_size = len(X_test)

    X_test = X_test[mask]
    y_test = y_test[mask]

    new_test_size = len(X_test)
    overlap_removed = original_test_size - new_test_size

    print(f"[INFO] Removed {overlap_removed} overlapping rows from TEST set")
    print(f"[INFO] Test set size: {original_test_size} -> {new_test_size}")


#define sigmoid function for binary PSO transfer
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


#initialize the swarm positions and velocities
def initialize_swarm(swarm_size, rng):
    #binary positions meaning whether feature is selected or not
    positions = rng.integers(0, 2, size=(swarm_size, num_features))

    #(prevent empty particle at initialization)
    for i in range(swarm_size):
        if np.sum(positions[i]) == 0:
            random_idx = rng.integers(0, num_features)
            positions[i, random_idx] = 1

    #velocity is continuous because PSO updates by movement
    velocities = rng.uniform(-1.0, 1.0, size=(swarm_size, num_features))

    return positions, velocities


#define the fitness function where one particle means one feature subset
#uses Random Forest as the base model (consistent with baseline)
#(fitness must use validation set instead of test set to avoid data leakage)
def fitness(individual):
    #prevent edge cases where no features are selected
    if np.sum(individual) == 0:
        return 0

    #define features selected are the one that has number 1 in string
    selected_features = np.where(individual == 1)[0]

    #select features that are only chosen
    X_tr = X_train[:, selected_features]
    X_va = X_val[:, selected_features]

    #random forest as the base IDS model - consistent with baseline comparison
    #cuml.accel will automatically redirect this to GPU if available
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    #train model on inner train dataset
    model.fit(X_tr, y_train)
    #predict it on the validation dataset
    y_pred = model.predict(X_va)

    #return the fitness value using validation labels and predicted labels
    #tries to maximize it
    return f1_score(y_val, y_pred, average="macro", zero_division=0)


#define one full PSO search for one config
def run_pso_config(config, config_index):
    swarm_size = config["swarm_size"]
    inertia_weight = config["w"]
    cognitive_weight = config["c1"]
    social_weight = config["c2"]
    config_name = config["name"]

    #(use config-specific rng so results are reproducible and independent)
    rng = np.random.default_rng(GLOBAL_SEED + config_index)

    #(each config gets its own wandb run so curves can be compared directly)
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

    #initialize the swarm here
    positions, velocities = initialize_swarm(swarm_size, rng)

    #personal best positions and scores
    personal_best_positions = positions.copy()
    personal_best_scores = np.array([fitness(particle) for particle in positions])

    #global best position and score
    global_best_idx = np.argmax(personal_best_scores)
    global_best_position = personal_best_positions[global_best_idx].copy()
    global_best_score = personal_best_scores[global_best_idx]

    #track total search time
    start_total = time.time()

    #start the iterations
    for iteration in range(ITERATIONS):
        #measure time from here
        start_iter = time.time()

        #store scores for logging
        current_scores = []

        #loop through all particles
        for i in range(swarm_size):
            #get random coefficients for update
            r1 = rng.random(num_features)
            r2 = rng.random(num_features)

            #velocity update rule
            velocities[i] = (
                inertia_weight * velocities[i]
                + cognitive_weight * r1 * (personal_best_positions[i] - positions[i])
                + social_weight * r2 * (global_best_position - positions[i])
            )

            #(clip velocity to avoid exploding movement)
            velocities[i] = np.clip(velocities[i], -6, 6)

            #binary PSO position update using sigmoid transfer
            probabilities = sigmoid(velocities[i])
            random_draw = rng.random(num_features)
            positions[i] = (random_draw < probabilities).astype(int)

            #(prevent empty particle after update)
            if np.sum(positions[i]) == 0:
                random_idx = rng.integers(0, num_features)
                positions[i, random_idx] = 1

            #evaluate current particle
            current_score = fitness(positions[i])
            current_scores.append(current_score)

            #update personal best if better
            if current_score > personal_best_scores[i]:
                personal_best_scores[i] = current_score
                personal_best_positions[i] = positions[i].copy()

                #update global best if better
                if current_score > global_best_score:
                    global_best_score = current_score
                    global_best_position = positions[i].copy()

        #get iteration time
        iter_time = time.time() - start_iter

        #print progress here
        print(
            f"[{config_name}] Iteration {iteration+1}/{ITERATIONS} "
            f"- Best Val F1: {global_best_score:.6f} - Time: {iter_time:.2f}s"
        )

        #let wandb track the best and average performance
        run.log({
            "iteration": iteration + 1,
            "best_val_f1": global_best_score,
            "avg_val_f1": float(np.mean(current_scores)),
            "iter_time": iter_time
        })

    total_search_time = time.time() - start_total

    #taking the best subset found by this config
    selected_features = np.where(global_best_position == 1)[0]

    #(retrain on the full original training set after PSO search is complete)
    X_tr_final = X_train_full[:, selected_features]
    X_te_final = X_test[:, selected_features]

    #train final model with Random Forest and measure the time
    start_train = time.time()
    final_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_tr_final, y_train_full)
    train_time = time.time() - start_train

    #prediction here
    start_test = time.time()
    y_pred = final_model.predict(X_te_final)
    test_time = time.time() - start_test

    #(also check train prediction to discuss overfitting more rigorously)
    y_train_pred = final_model.predict(X_tr_final)

    #finally compute the metrices that are needed
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    #(train metrics for overfitting check)
    train_acc = accuracy_score(y_train_full, y_train_pred)
    train_prec = precision_score(y_train_full, y_train_pred, average="macro", zero_division=0)
    train_rec = recall_score(y_train_full, y_train_pred, average="macro", zero_division=0)
    train_f1 = f1_score(y_train_full, y_train_pred, average="macro", zero_division=0)
    train_bal_acc = balanced_accuracy_score(y_train_full, y_train_pred)

    #breakdown of matrix
    cm = confusion_matrix(y_test, y_pred)

    #print the classification report
    print(f"\n===== [{config_name}] CLASSIFICATION REPORT =====\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    #print the results of this config
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

    #(log final metrics of this config so each config run is self-contained)
    run.log({
        "overlap_count_detected": overlap_count,
        "overlap_removed_from_test": overlap_removed,

        "best_validation_f1": global_best_score,
        "num_selected_features": len(selected_features),

        "train_accuracy": train_acc,
        "train_precision": train_prec,
        "train_recall": train_rec,
        "train_f1": train_f1,
        "train_balanced_accuracy": train_bal_acc,

        "final_accuracy": acc,
        "final_precision": prec,
        "final_recall": rec,
        "final_f1": f1,
        "final_balanced_accuracy": bal_acc,

        "train_time": train_time,
        "test_time": test_time,
        "total_search_time": total_search_time
    })

    #(log confusion matrix if wandb available)
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
        "config_name": config_name,
        "swarm_size": swarm_size,
        "w": inertia_weight,
        "c1": cognitive_weight,
        "c2": social_weight,
        "best_validation_f1": global_best_score,
        "selected_features": selected_features,
        "num_selected_features": len(selected_features),
        "search_time": total_search_time,

        "train_accuracy": train_acc,
        "train_precision": train_prec,
        "train_recall": train_rec,
        "train_f1": train_f1,
        "train_balanced_accuracy": train_bal_acc,

        "final_accuracy": acc,
        "final_precision": prec,
        "final_recall": rec,
        "final_f1": f1,
        "final_balanced_accuracy": bal_acc,

        "train_time": train_time,
        "test_time": test_time,
        "confusion_matrix": cm
    }


#(run all configs and choose the best one by validation score only)
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

#(choose best config only by validation performance)
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