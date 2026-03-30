#the core of genetic algorithm is to select amount of initial cadidates
#evaluates and apply fitness function to each candidate to be the parent
#through iterations crossover and mutation until a local optima is found
#(high score version: one wandb run per config, validation for search, test only once, safe overlap handling)

#allow gpu run - graceful fallback if cuML not available
from env_setup import GPU_AVAILABLE, WANDB_AVAILABLE, DATA_DIR, TARGET_COLUMN, init_wandb

#get pandas to read the csv
import pandas as pd

#get numpy for array and matrix operations
import numpy as np

#python to generate random values which is used for
#1. random selection 2.crossover 3.mutation
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

#(split train into train_inner and val_inner so GA search never touches the test set)
from sklearn.model_selection import train_test_split


#(global seed for reproducibility)
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

#define number of evolutions that will happen
GENERATIONS = 15
#probs of combining 2 parents
CROSSOVER_RATE = 0.8
#tournament size indicates amount of gene compete during selection
TOURNAMENT_SIZE = 3
#(validation split ratio used only inside training data for GA fitness)
VAL_SIZE = 0.20

#(hyperparameter study for GA)
GA_CONFIGS = [
    {"name": "GA_RAW_pop20_mut005", "population_size": 20, "mutation_rate": 0.05},
    {"name": "GA_RAW_pop20_mut010", "population_size": 20, "mutation_rate": 0.10},
    {"name": "GA_RAW_pop20_mut020", "population_size": 20, "mutation_rate": 0.20},
    {"name": "GA_RAW_pop50_mut005", "population_size": 50, "mutation_rate": 0.05},
    {"name": "GA_RAW_pop50_mut010", "population_size": 50, "mutation_rate": 0.10},
    {"name": "GA_RAW_pop50_mut020", "population_size": 50, "mutation_rate": 0.20},
]

#get the train raw dataset
train_df = pd.read_csv(f"{DATA_DIR}/train_multiclass.csv")
#get the test raw dataset
test_df = pd.read_csv(f"{DATA_DIR}/test_multiclass.csv")

#remove the label column for the machine learning
X_train_full = train_df.drop(columns=[TARGET_COLUMN]).values
#get the values of target using numpy
y_train_full = train_df[TARGET_COLUMN].values

#remove the label column for testing the model
X_test = test_df.drop(columns=[TARGET_COLUMN]).values
#same as train data here
y_test = test_df[TARGET_COLUMN].values

#(split the original training set into inner-train and validation for GA search only)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=VAL_SIZE,
    random_state=GLOBAL_SEED,
    stratify=y_train_full
)

#define number of features as the num of column in train datase
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


#(check overlap once before running GA so no accidental leakage slips in)
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


#first step of GA which is initialization of population
def initialize_population(pop_size, rng):
    #here ensures binary generation from 0<= x <2, only 0 or 1
    #random ensures the GA with random spot scattered
    population = rng.integers(0, 2, (pop_size, num_features))

    #(prevent empty individual at initialization)
    for i in range(pop_size):
        if np.sum(population[i]) == 0:
            random_idx = rng.integers(0, num_features)
            population[i, random_idx] = 1

    return population


#define the fitness function where individual / one solution
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


#define the tournament selection / selecting parents here
def tournament_selection(population, fitness_scores, pop_size, rng):
    #store the selected feature
    selected = []
    #you want to loop until new population is full
    for _ in range(pop_size):
        #pick random individuals as candidates dependent on tournament size
        candidates = rng.choice(pop_size, size=TOURNAMENT_SIZE, replace=False)
        #pick the best candidate with highest fitness
        best = max(candidates, key=lambda idx: fitness_scores[idx])
        #now the best are selected for the next generation
        selected.append(population[best].copy())
    return np.array(selected)


#define crossover function between 2 parents
def crossover(parent1, parent2, pop_rng):
    #only perform crossover with some probability
    if pop_rng.random() < CROSSOVER_RATE:
        #choose the crossover point
        point = pop_rng.integers(1, num_features)
        #swapping the 2 parts of parents
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()


#define mutation
def mutate(individual, mutation_rate, pop_rng):
    #in each features depend on mutation rate to change
    for i in range(num_features):
        if pop_rng.random() < mutation_rate:
            individual[i] = 1 - individual[i]

    #(prevent mutation from creating empty individual)
    if np.sum(individual) == 0:
        random_idx = pop_rng.integers(0, num_features)
        individual[random_idx] = 1

    return individual


#(run one full GA search for one config and return the best result)
def run_ga_config(config, config_index):
    pop_size = config["population_size"]
    mutation_rate = config["mutation_rate"]
    config_name = config["name"]

    #(use config-specific rng so results are reproducible and independent)
    rng = np.random.default_rng(GLOBAL_SEED + config_index)

    #(each config gets its own wandb run so curves can be compared directly)
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

    #FULL GA EVOLUTION
    population = initialize_population(pop_size, rng)

    #track best solution so far
    best_individual = None
    best_fitness = -1
    best_generation = -1

    config_start_total = time.time()

    #start the evolution
    for gen in range(GENERATIONS):
        #measure time from here
        start_gen = time.time()

        #evaluate all individuals
        fitness_scores = np.array([fitness(ind) for ind in population])

        #the best in generations
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]

        #update the global best here
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = population[gen_best_idx].copy()
            best_generation = gen + 1

        #print out the progress
        gen_time = time.time() - start_gen
        print(
            f"[{config_name}] Generation {gen+1}/{GENERATIONS} "
            f"- Best Val F1: {gen_best_fitness:.6f} - Time: {gen_time:.2f}s"
        )

        #let wandb track the best and average performance
        run.log({
            "generation": gen + 1,
            "best_val_f1": gen_best_fitness,
            "avg_val_f1": float(np.mean(fitness_scores)),
            "gen_time": gen_time
        })

        #choose parents here
        selected_population = tournament_selection(population, fitness_scores, pop_size, rng)
        #prepare next generation
        next_population = []

        #(elitism: keep the best solution so it is not lost in the next generation)
        next_population.append(best_individual.copy())

        #loop again
        while len(next_population) < pop_size:
            #select parent
            parent1 = selected_population[rng.integers(0, pop_size)]
            parent2 = selected_population[rng.integers(0, pop_size)]

            #crossover here
            child1, child2 = crossover(parent1, parent2, rng)

            #mutate here -add randomness
            child1 = mutate(child1, mutation_rate, rng)
            child2 = mutate(child2, mutation_rate, rng)

            #add to the next generation
            next_population.append(child1)
            if len(next_population) < pop_size:
                next_population.append(child2)

        #replace the population
        population = np.array(next_population)

    total_search_time = time.time() - config_start_total

    #taking the best solution and train the model
    selected_features = np.where(best_individual == 1)[0]

    #(retrain on the full original training set after GA search is complete)
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

    #(log final metrics of this config so each config run is self-contained)
    run.log({
        "overlap_count_detected": overlap_count,
        "overlap_removed_from_test": overlap_removed,

        "best_validation_f1": best_fitness,
        "best_generation": best_generation,
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
        "population_size": pop_size,
        "mutation_rate": mutation_rate,
        "best_validation_f1": best_fitness,
        "best_generation": best_generation,
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

print("\n===== START GA HYPERPARAMETER SEARCH =====")
for idx, config in enumerate(GA_CONFIGS, start=1):
    print(
        f"\n===== RUNNING CONFIG {idx}/{len(GA_CONFIGS)} : "
        f"{config['name']} "
        f"(pop={config['population_size']}, mut={config['mutation_rate']}) ====="
    )

    result = run_ga_config(config, idx)
    all_results.append(result)

#(choose best config only by validation performance)
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