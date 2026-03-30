#random forest is build by treating datasetnas matrix
#its core is to form loads of decision tree instead of independent tree getting results
#1. bootstrap sampling is to create diverse in rows in instance from 1,2,3,4,5 -> 2,2,3,5,1
#2. random feature sampling where picking few subset of features for square_root(total features)
#3. using the decision tree logic to come out with a label/results for each tree
#4. final voting by collecting the most amount of category to be its answer

#allow gpu to run in this case
from env_setup import GPU_AVAILABLE, WANDB_AVAILABLE, DATA_DIR, TARGET_COLUMN, init_wandb

#import pandas for reading csv
import pandas as pd

#import numpy for arrays and matrix operations
import numpy as np

#import RF from library will all default values
from sklearn.ensemble import RandomForestClassifier

#import confusion matrix for self view if the model is being bias due to dataset
#as well as computing false true rate
from sklearn.metrics import confusion_matrix

#call for evaluation of metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    balanced_accuracy_score,
)

#import time to record time taken for the process, needed for critetia
import time


#get the train raw dataset
train_df = pd.read_csv(f"{DATA_DIR}/train_multiclass.csv")
#get the test raw dataset
test_df = pd.read_csv(f"{DATA_DIR}/test_multiclass.csv")

#remove the label column leaving the rest
X_train = train_df.drop(columns=[TARGET_COLUMN]).values
#turn the label column into array by the help of Numpy
y_train = train_df[TARGET_COLUMN].values

#the same procedure of training set apply to test set
X_test = test_df.drop(columns=[TARGET_COLUMN]).values
y_test = test_df[TARGET_COLUMN].values


#strict overlap detection so baseline uses the same test protocol style as metaheuristics
def exact_overlap_count(X_a, y_a, X_b, y_b):
    #combine features and labels together so overlap check is strict
    a = np.column_stack([X_a, y_a])
    b = np.column_stack([X_b, y_b])

    #convert each row into bytes for exact set comparison
    set_a = {row.tobytes() for row in a}
    set_b = {row.tobytes() for row in b}

    return len(set_a.intersection(set_b))


#check overlap once before running RF so no accidental leakage slips in
overlap_count = exact_overlap_count(X_train, y_train, X_test, y_test)
print("\n===== DATA SANITY CHECK =====")
print("Exact Train-Test Overlap Count :", overlap_count)

overlap_removed = 0

if overlap_count != 0:
    print(f"\n[WARNING] Found {overlap_count} overlapping rows between train and test.")

    #combine features and labels together so overlap check is strict
    a = np.column_stack([X_train, y_train])
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


#initialize wandb run
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

#import the model classifier since this is a categorical identification not continues numbes
#n_jobs=-1 lets CPU parallelism work if GPU accel does not take over
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

#start the timer
start = time.time()

#do bootstrap from sample and split it into 100 trees by each choosing subset of features
rf.fit(X_train, y_train)

#record the time for training
train_time = time.time() - start

#predict on training set too so overfitting can be discussed more rigorously
start = time.time()
y_train_pred = rf.predict(X_train)
train_predict_time = time.time() - start

#start timer again for prediction
start = time.time()

#now try to predict labels using X_test
y_pred = rf.predict(X_test)

#record time for prediction
predict_time = time.time() - start

#have the total runtime here
total_time = train_time + predict_time

#get it by accuracy = correct_predictions / total_predictions
train_accuracy = accuracy_score(y_train, y_train_pred)

#brutally and straightly tell if the model is right/how often the model is right
train_precision = precision_score(y_train, y_train_pred, average="macro", zero_division=0)

#looking how many were correctly detected
train_recall = recall_score(y_train, y_train_pred, average="macro", zero_division=0)

#f1 is used to balance between precision and recall
train_f1 = f1_score(y_train, y_train_pred, average="macro", zero_division=0)

#(balanced accuracy is useful for imbalanced multiclass dataset)
train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)

#get it by accuracy = correct_predictions / total_predictions
accuracy = accuracy_score(y_test, y_pred)

#brutally and straightly tell if the model is right/how often the model is right
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)

#looking how many were correctly detected
recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

#f1 is used to balance between precision and recall
f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

#balanced accuracy is useful for imbalanced multiclass dataset
balanced_acc = balanced_accuracy_score(y_test, y_pred)

#generate the confusion matrix using the prediction values and real labels
cm = confusion_matrix(y_test, y_pred)


#plot out the confusion matrix for wandb
if WANDB_AVAILABLE:
    import wandb
    run.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_test,
            preds=y_pred
        )
    })


#get the false posive and the true negative
#usually use for binary hence we can only do approximation on multi label
print("\nPer-class metrics from confusion matrix:")

num_classes = cm.shape[0]

for i in range(num_classes):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0  # recall

    print(f"\nClass {i}:")
    print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"FPR={fpr:.6f}, TPR(Recall)={tpr:.6f}")


#compute how many features was selected
n_features = X_train.shape[1]

#now get the train results first
print("\n===== TRAIN RESULTS =====")
print("Train Accuracy :", train_accuracy)
print("Train Precision:", train_precision)
print("Train Recall   :", train_recall)
print("Train F1 Score :", train_f1)
print("Train Balanced Accuracy :", train_balanced_accuracy)
print("Train Predict Time:", train_predict_time)

#now get the test results we needed in here
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

#print overlap handling info so baseline protocol is transparent
print("\n===== OVERLAP HANDLING =====")
print("Overlap Count Detected :", overlap_count)
print("Overlap Removed From Test :", overlap_removed)
print("Final Test Size :", len(y_test))

# log metrics to wandb
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

#print the classification performance
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

#print train-test gap for quick overfitting check
print("\n===== TRAIN-TEST GAP =====")
print("Accuracy Gap :", train_accuracy - accuracy)
print("Precision Gap:", train_precision - precision)
print("Recall Gap   :", train_recall - recall)
print("F1 Gap       :", train_f1 - f1)
print("Balanced Acc Gap :", train_balanced_accuracy - balanced_acc)

#finish wandb run
run.finish()