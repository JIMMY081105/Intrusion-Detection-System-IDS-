# File:    dataset_prepare.py
# Purpose: Full preprocessing pipeline for the CICIDS2017 multiclass intrusion detection dataset.
#          Merges raw CSVs, cleans labels, removes exact duplicates, performs a stratified
#          80/20 train-test split, applies median imputation and standard scaling, and fits PCA
#          retaining 80% of variance. All transformations are fit on the training set only.
# Input:   data/raw/*.csv  (raw CICIDS2017 CSV files)
# Output:  data/processed/ (train/test splits, PCA splits, audit logs, label mapping)

import os
import glob
import json
import hashlib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MERGED_RAW_PATH       = os.path.join(OUTPUT_DIR, "merged_raw_multiclass.csv")
DEDUPED_FULL_PATH     = os.path.join(OUTPUT_DIR, "deduplicated_full_multiclass.csv")
TRAIN_BASIC_PATH      = os.path.join(OUTPUT_DIR, "train_multiclass.csv")
TEST_BASIC_PATH       = os.path.join(OUTPUT_DIR, "test_multiclass.csv")
TRAIN_PCA_PATH        = os.path.join(OUTPUT_DIR, "train_pca_multiclass.csv")
TEST_PCA_PATH         = os.path.join(OUTPUT_DIR, "test_pca_multiclass.csv")
LABEL_MAP_PATH        = os.path.join(OUTPUT_DIR, "label_mapping.csv")
PCA_VARIANCE_PATH     = os.path.join(OUTPUT_DIR, "pca_variance_multiclass.csv")
PCA_LOADINGS_PATH     = os.path.join(OUTPUT_DIR, "pca_feature_loadings_multiclass.csv")
PCA_FEATURE_USAGE_PATH = os.path.join(OUTPUT_DIR, "pca_feature_usage_multiclass.csv")
AUDIT_SUMMARY_JSON    = os.path.join(OUTPUT_DIR, "dataset_audit_summary.json")
TRAIN_CLASS_DIST_PATH = os.path.join(OUTPUT_DIR, "train_class_distribution.csv")
TEST_CLASS_DIST_PATH  = os.path.join(OUTPUT_DIR, "test_class_distribution.csv")
TRAIN_DUPLICATES_PATH = os.path.join(OUTPUT_DIR, "train_duplicate_rows.csv")
TEST_DUPLICATES_PATH  = os.path.join(OUTPUT_DIR, "test_duplicate_rows.csv")
OVERLAP_ROWS_PATH     = os.path.join(OUTPUT_DIR, "train_test_exact_overlap_rows.csv")

# ─── Configuration ────────────────────────────────────────────────────────────

TEST_SIZE           = 0.20
RANDOM_STATE        = 42
PCA_VARIANCE_TO_KEEP = 0.80
USAGE_THRESHOLD     = 0.05

# These columns are identifiers, not network traffic features.
DROP_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp"
]

LABEL_COLUMN  = "Label"
TARGET_COLUMN = "Target"


# ─── Helper Functions ─────────────────────────────────────────────────────────

def stable_row_hash(df: pd.DataFrame, chunk_size: int = 50000) -> pd.Series:
    """Compute a stable MD5 hash per row for duplicate and overlap detection."""
    parts = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        row_strings = chunk.astype(str).agg("||".join, axis=1)
        hashes = row_strings.apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
        parts.append(hashes)
        del chunk, row_strings
    return pd.concat(parts)


def save_class_distribution(y: pd.Series, label_mapping_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Save a CSV report showing sample count and percentage for each class."""
    dist = y.value_counts().sort_index().rename_axis(TARGET_COLUMN).reset_index(name="Count")
    dist = dist.merge(label_mapping_df, on=TARGET_COLUMN, how="left")
    dist["Percentage"] = dist["Count"] / dist["Count"].sum()
    dist.to_csv(output_path, index=False)
    return dist


def save_duplicate_rows(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Save all duplicate rows (both copies) for auditing purposes."""
    dup_mask = df.duplicated(keep=False)
    dup_df = df.loc[dup_mask].copy()
    if not dup_df.empty:
        dup_df["_duplicate_group_hash"] = stable_row_hash(dup_df)
    dup_df.to_csv(output_path, index=False)
    return dup_df


def build_feature_usage(pca, feature_names, usage_threshold):
    """Build a table showing how much each original feature contributes to the retained PCs."""
    pca_columns = [f"PC{i+1}" for i in range(pca.n_components_)]
    loadings_abs = np.abs(pca.components_)

    max_loading   = loadings_abs.max(axis=0)
    total_loading = loadings_abs.sum(axis=0)
    num_pcs_above_threshold = (loadings_abs > usage_threshold).sum(axis=0)

    feature_usage = pd.DataFrame({
        "Feature":              feature_names,
        "Max_Abs_Loading":      max_loading,
        "Total_Abs_Loading":    total_loading,
        "Num_PCs_Above_Threshold": num_pcs_above_threshold
    })

    feature_usage["Used_By_PCA"] = feature_usage["Max_Abs_Loading"] > usage_threshold

    def explain_usage(row):
        if row["Used_By_PCA"]:
            return "Used: feature has meaningful contribution to at least one retained principal component."
        return "Not used significantly: contribution is near zero across retained principal components."

    feature_usage["Reason"] = feature_usage.apply(explain_usage, axis=1)

    feature_usage = feature_usage.sort_values(
        by=["Used_By_PCA", "Max_Abs_Loading", "Total_Abs_Loading"],
        ascending=[False, False, False]
    )

    return feature_usage, pca_columns


# ─── Step 1: Load Raw CSV Files ───────────────────────────────────────────────

print(f"[INFO] Looking for raw CSV files in: {RAW_DATA_PATH}")
csv_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.csv"))
print(f"CSV files found: {len(csv_files)}")

if len(csv_files) == 0:
    raise FileNotFoundError(
        f"No CSV files found in: {RAW_DATA_PATH}\n"
        "Please download the CICIDS2017 dataset and place the CSV files in data/raw/."
    )

dfs = []
for file in csv_files:
    print(f"Loading: {file}")
    df = pd.read_csv(file, low_memory=False, encoding="utf-8", encoding_errors="replace")
    df.columns = df.columns.str.strip()
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
del dfs
data.columns = data.columns.str.strip()

raw_merged_shape = data.shape
print(f"Merged dataset shape: {raw_merged_shape}")
data.to_csv(MERGED_RAW_PATH, index=False)

# ─── Step 2: Clean Labels ─────────────────────────────────────────────────────

if LABEL_COLUMN not in data.columns:
    raise KeyError(f"Column '{LABEL_COLUMN}' not found in merged dataset.")

data[LABEL_COLUMN] = data[LABEL_COLUMN].astype(str).str.strip()

# Normalise encoding artefacts that appear in some CICIDS2017 label strings.
data[LABEL_COLUMN] = (
    data[LABEL_COLUMN]
    .str.replace("â€"", "-", regex=False)
    .str.replace("–", "-", regex=False)
)

# Drop identifier columns that are not traffic features.
existing_drop_columns = [col for col in DROP_COLUMNS if col in data.columns]
if existing_drop_columns:
    data = data.drop(columns=existing_drop_columns)
    print(f"Dropped non-feature columns: {existing_drop_columns}")

# ─── Step 3: Encode Labels ────────────────────────────────────────────────────

unique_labels = sorted(data[LABEL_COLUMN].unique())
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
data[TARGET_COLUMN] = data[LABEL_COLUMN].map(label_to_id)

if data[TARGET_COLUMN].isna().any():
    raise ValueError("Some labels were not mapped correctly.")

label_mapping_df = pd.DataFrame({
    TARGET_COLUMN:   list(label_to_id.values()),
    "Original_Label": list(label_to_id.keys())
}).sort_values(TARGET_COLUMN)

label_mapping_df.to_csv(LABEL_MAP_PATH, index=False)

data = data.drop(columns=[LABEL_COLUMN])
data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

# ─── Step 4: Handle Infinities ────────────────────────────────────────────────

y = data[TARGET_COLUMN].copy()
X = data.drop(columns=[TARGET_COLUMN])
del data

non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    raise ValueError(f"Non-numeric feature columns found: {non_numeric_cols}")

numeric_cols = X.columns.tolist()
inf_count_before_split = np.isinf(X[numeric_cols]).sum().sum()
print(f"Total inf values before split: {inf_count_before_split}")

X.replace([np.inf, -np.inf], np.nan, inplace=True)

nan_count_after_inf_replace    = X.isna().sum().sum()
rows_with_nan_after_inf_replace = X.isna().any(axis=1).sum()

print(f"Total NaN after inf replacement: {nan_count_after_inf_replace}")
print(f"Rows containing NaN after inf replacement: {rows_with_nan_after_inf_replace}")

# ─── Step 5: Deduplicate BEFORE Split ────────────────────────────────────────

# Deduplication is done before the split to prevent duplicate rows from appearing
# in both train and test, which would inflate test performance.
X[TARGET_COLUMN] = y.values
full_df = X
del X, y

rows_before_dedup         = len(full_df)
exact_duplicate_rows_before = int(full_df.duplicated().sum())

full_df = full_df.drop_duplicates().reset_index(drop=True)

rows_after_dedup      = len(full_df)
rows_removed_by_dedup = rows_before_dedup - rows_after_dedup

print(f"Rows before deduplication: {rows_before_dedup}")
print(f"Exact duplicate rows before deduplication: {exact_duplicate_rows_before}")
print(f"Rows removed by deduplication: {rows_removed_by_dedup}")
print(f"Rows after deduplication: {rows_after_dedup}")

full_df.to_csv(DEDUPED_FULL_PATH, index=False)

# ─── Step 6: Train-Test Split ────────────────────────────────────────────────

X = full_df.drop(columns=[TARGET_COLUMN])
y = full_df[TARGET_COLUMN].astype(int)
del full_df

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y     # preserve class proportions in both splits
)
del X, y

print(f"Train feature shape: {X_train.shape}")
print(f"Test feature shape: {X_test.shape}")

# ─── Step 7: Imputation ───────────────────────────────────────────────────────

# Fit on train only — transform is applied to both train and test using train statistics.
imputer = SimpleImputer(strategy="median")

train_columns = X_train.columns
train_index   = X_train.index
test_columns  = X_test.columns
test_index    = X_test.index

X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=train_columns,
    index=train_index
)
del X_train

X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=test_columns,
    index=test_index
)
del X_test

train_remaining_nan = X_train_imputed.isna().sum().sum()
test_remaining_nan  = X_test_imputed.isna().sum().sum()

if train_remaining_nan != 0 or test_remaining_nan != 0:
    raise ValueError(
        f"NaN remains after imputation. Train NaN={train_remaining_nan}, Test NaN={test_remaining_nan}"
    )

train_inf_after = np.isinf(X_train_imputed).sum().sum()
test_inf_after  = np.isinf(X_test_imputed).sum().sum()

if train_inf_after != 0 or test_inf_after != 0:
    raise ValueError(
        f"Inf remains after imputation. Train inf={train_inf_after}, Test inf={test_inf_after}"
    )

# ─── Step 8: Save Basic Train/Test ───────────────────────────────────────────

X_train_imputed[TARGET_COLUMN] = y_train.values
X_test_imputed[TARGET_COLUMN]  = y_test.values

X_train_imputed.to_csv(TRAIN_BASIC_PATH, index=False)
X_test_imputed.to_csv(TEST_BASIC_PATH, index=False)

train_basic_df = X_train_imputed
test_basic_df  = X_test_imputed

# Verify no duplicates crept back in after the split.
train_dup_count = int(train_basic_df.duplicated().sum())
test_dup_count  = int(test_basic_df.duplicated().sum())

print(f"Duplicate rows inside TRAIN after dedup: {train_dup_count}")
print(f"Duplicate rows inside TEST after dedup: {test_dup_count}")

save_duplicate_rows(train_basic_df, TRAIN_DUPLICATES_PATH)
save_duplicate_rows(test_basic_df,  TEST_DUPLICATES_PATH)

# Check for exact overlap between train and test rows.
train_hashes = stable_row_hash(train_basic_df)
test_hashes  = stable_row_hash(test_basic_df)

overlap_hashes = set(train_hashes).intersection(set(test_hashes))
overlap_count  = len(overlap_hashes)

print(f"Exact train-test overlap row count after dedup: {overlap_count}")

if overlap_count > 0:
    train_overlap = train_basic_df[train_hashes.isin(overlap_hashes)].copy()
    train_overlap["_row_hash"] = train_hashes[train_hashes.isin(overlap_hashes)]
    train_overlap["_source"]   = "train"
    test_overlap = test_basic_df[test_hashes.isin(overlap_hashes)].copy()
    test_overlap["_row_hash"]  = test_hashes[test_hashes.isin(overlap_hashes)]
    test_overlap["_source"]    = "test"
    overlap_rows = pd.concat([train_overlap, test_overlap], ignore_index=True)
    del train_overlap, test_overlap
else:
    overlap_rows = pd.DataFrame(columns=list(train_basic_df.columns) + ["_row_hash", "_source"])

del train_hashes, test_hashes
overlap_rows.to_csv(OVERLAP_ROWS_PATH, index=False)

save_class_distribution(y_train, label_mapping_df, TRAIN_CLASS_DIST_PATH)
save_class_distribution(y_test,  label_mapping_df, TEST_CLASS_DIST_PATH)

del overlap_rows

# ─── Step 9: Scaling and PCA ─────────────────────────────────────────────────

feature_cols        = [c for c in train_basic_df.columns if c != TARGET_COLUMN]
feature_names       = feature_cols
num_original_features = len(feature_cols)
train_imputed_index = train_basic_df.index
test_imputed_index  = test_basic_df.index
train_basic_columns = list(train_basic_df.columns)
test_basic_columns  = list(test_basic_df.columns)
train_basic_shape   = train_basic_df.shape
test_basic_shape    = test_basic_df.shape

# StandardScaler fit on train only to prevent leakage of test statistics.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(train_basic_df[feature_cols])
X_test_scaled  = scaler.transform(test_basic_df[feature_cols])

del train_basic_df, test_basic_df

# PCA fit on train only, retaining 80% of cumulative explained variance.
pca = PCA(n_components=PCA_VARIANCE_TO_KEEP, random_state=RANDOM_STATE)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

print("PCA complete")
print(f"Original feature count: {num_original_features}")
print(f"PCA feature count: {X_train_pca.shape[1]}")
print(f"Total explained variance retained: {pca.explained_variance_ratio_.sum():.6f}")

# ─── Step 10: Save PCA Outputs ───────────────────────────────────────────────

pca_columns = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]

train_pca_df = pd.DataFrame(X_train_pca, columns=pca_columns, index=train_imputed_index)
train_pca_df[TARGET_COLUMN] = y_train.values

test_pca_df = pd.DataFrame(X_test_pca, columns=pca_columns, index=test_imputed_index)
test_pca_df[TARGET_COLUMN] = y_test.values

train_pca_df.to_csv(TRAIN_PCA_PATH, index=False)
test_pca_df.to_csv(TEST_PCA_PATH,   index=False)

variance_df = pd.DataFrame({
    "Principal_Component":          pca_columns,
    "Explained_Variance":           pca.explained_variance_ratio_,
    "Cumulative_Explained_Variance": np.cumsum(pca.explained_variance_ratio_)
})
variance_df.to_csv(PCA_VARIANCE_PATH, index=False)

loadings_df = pd.DataFrame(
    pca.components_.T,
    columns=pca_columns,
    index=feature_names
)
loadings_df.to_csv(PCA_LOADINGS_PATH)

feature_usage_df, _ = build_feature_usage(
    pca=pca,
    feature_names=feature_names,
    usage_threshold=USAGE_THRESHOLD
)
feature_usage_df.to_csv(PCA_FEATURE_USAGE_PATH, index=False)

# ─── Step 11: Validation and Audit ───────────────────────────────────────────

basic_columns_match = train_basic_columns == test_basic_columns
pca_columns_match   = list(train_pca_df.columns) == list(test_pca_df.columns)

if not basic_columns_match:
    raise ValueError("Basic train/test columns do not match.")
if not pca_columns_match:
    raise ValueError("PCA train/test columns do not match.")

audit_summary = {
    "raw_files_found":                            len(csv_files),
    "raw_merged_dataset_shape":                   [int(raw_merged_shape[0]), int(raw_merged_shape[1])],
    "rows_before_deduplication":                  int(rows_before_dedup),
    "exact_duplicate_rows_before_deduplication":  int(exact_duplicate_rows_before),
    "rows_removed_by_deduplication":              int(rows_removed_by_dedup),
    "rows_after_deduplication":                   int(rows_after_dedup),
    "train_basic_shape":                          [int(train_basic_shape[0]), int(train_basic_shape[1])],
    "test_basic_shape":                           [int(test_basic_shape[0]), int(test_basic_shape[1])],
    "train_pca_shape":                            [int(train_pca_df.shape[0]), int(train_pca_df.shape[1])],
    "test_pca_shape":                             [int(test_pca_df.shape[0]), int(test_pca_df.shape[1])],
    "num_original_features":                      int(num_original_features),
    "num_pca_features":                           int(X_train_pca.shape[1]),
    "pca_explained_variance_sum":                 float(pca.explained_variance_ratio_.sum()),
    "inf_count_before_split":                     int(inf_count_before_split),
    "nan_count_after_inf_replace_before_split":   int(nan_count_after_inf_replace),
    "rows_with_nan_after_inf_replace_before_split": int(rows_with_nan_after_inf_replace),
    "train_remaining_nan_after_imputation":       int(train_remaining_nan),
    "test_remaining_nan_after_imputation":        int(test_remaining_nan),
    "train_inf_after_imputation":                 int(train_inf_after),
    "test_inf_after_imputation":                  int(test_inf_after),
    "train_duplicate_row_count_after_dedup":      int(train_dup_count),
    "test_duplicate_row_count_after_dedup":       int(test_dup_count),
    "train_test_exact_overlap_count_after_dedup": int(overlap_count),
    "basic_train_test_columns_match":             bool(basic_columns_match),
    "pca_train_test_columns_match":               bool(pca_columns_match),
    "label_count":                                int(len(label_mapping_df)),
    "labels":                                     label_mapping_df.to_dict(orient="records")
}

with open(AUDIT_SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(audit_summary, f, indent=2, ensure_ascii=False)

print(f"Audit summary saved to: {AUDIT_SUMMARY_JSON}")

print("\n========== FINAL SUMMARY ==========")
print("Exact duplicate full rows removed BEFORE split.")
print("Basic train/test CSVs created from the SAME split as PCA.")
print("Median imputation fitted on TRAIN only.")
print("Scaling fitted on TRAIN only.")
print("PCA fitted on TRAIN only.")
print(f"Rows removed by deduplication: {rows_removed_by_dedup}")
print(f"Train duplicate rows after dedup: {train_dup_count}")
print(f"Test duplicate rows after dedup: {test_dup_count}")
print(f"Exact train-test overlap rows after dedup: {overlap_count}")
print(f"PCA retained {X_train_pca.shape[1]} components with explained variance = {pca.explained_variance_ratio_.sum():.6f}")
print("All done.")
