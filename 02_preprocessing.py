# =============================================================================
# 02_preprocessing.py — Data Preparation
# =============================================================================
"""WHAT IS THIS FILE?
This is the second step of the project: preparing the data so it is ready
to be fed into a classifier. This includes cleaning, splitting into
train/test sets, and normalizing features.

WHY DO WE DO THIS BEFORE MODELING?
Every classifier we use later (kNN, SVM, Random Forest) depends on this
file producing a train/test split and normalization that is done honestly —
meaning no information from the test set leaks into training. Lecture 3b
calls this the most common, most damaging mistake in ML evaluation."""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# =============================================================================
# SECTION 2: LOAD THE DATA
# =============================================================================
# Same loading procedure as 01_eda.py — column names supplied manually since
# the file has no header row.

column_names = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1",
    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"
]

df = pd.read_csv(
    os.path.join("data", "magic04.data"),
    header=None,
    names=column_names
)

print("DATA LOADED SUCCESSFULLY")
print(f"Shape before cleaning: {df.shape}")


# =============================================================================
# SECTION 3: REMOVE DUPLICATE ROWS
# =============================================================================
"""01_eda.py checked for duplicate rows and reported the count. Here we
actually remove them, BEFORE splitting into train/test.

WHY BEFORE THE SPLIT?
If a duplicate row exists and we split AFTER removing duplicates, there is
no risk. But if we split BEFORE removing duplicates, a duplicate pair could
end up with one copy in training and the other in test — meaning the model
effectively gets tested on a point it already saw during training. This
breaks the train/test independence rule from Lecture 3b/7."""

n_before = df.shape[0]
df = df.drop_duplicates()
n_after = df.shape[0]

print(f"\n--- Duplicate removal ---")
print(f"Rows before: {n_before}")
print(f"Rows after: {n_after}")
print(f"Removed: {n_before - n_after}")

"""WHAT WE FOUND:
01_eda.py reported 0 (or very few) duplicate rows for this dataset, so we
expect little to no change here. This step is mainly a safeguard and a
demonstration that we checked for this issue before splitting."""


# =============================================================================
# SECTION 4: TRAIN / TEST SPLIT
# =============================================================================
"""We split the data into a training set (80%) and a test set (20%).

WHY 80/20?
With 19020 samples, an 80/20 split still leaves a large test set (~3800
samples) — large enough to give a stable, reliable estimate of performance,
while keeping most of the data available for training (Lecture 7's
guidance on test set size).

WHY STRATIFIED?
01_eda.py found the classes are imbalanced (~65% gamma, ~35% hadron). A
plain random split could, by chance, end up with a different class ratio
in the test set than in the training set. Stratifying the split forces
both the training set and the test set to preserve the same ~65/35 ratio
found in the full dataset. This matches Lecture 3b's warning about hidden
connections and representativeness between train and test data."""

X = df[[col for col in df.columns if col != "class"]]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,          # preserves class ratio in both sets
    random_state=42       # reproducible split
)

print(f"\n--- Train/test split ---")
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

print("\nClass ratio in training set:")
print(y_train.value_counts(normalize=True).round(3))

print("\nClass ratio in test set:")
print(y_test.value_counts(normalize=True).round(3))

"""WHAT TO LOOK FOR:
Both the training set and the test set should show approximately the same
class proportions as the full dataset (~65% gamma, ~35% hadron). If the
proportions differ noticeably, the split did not stratify correctly."""

"""IMPORTANT RULE GOING FORWARD (Lecture 10 — "data snooping"):
From this point on, X_test and y_test are LOCKED AWAY. We do not look at
them again, plot them, or use them to make any decision until the very
final evaluation in 06_evaluation.py. Any hyperparameter tuning in
03/04/05 will use cross-validation on X_train/y_train only."""


# =============================================================================
# SECTION 5: NORMALIZE FEATURES
# =============================================================================
"""kNN and SVM both rely on distances between points (Lecture 3a kNN,
Lecture 9 SVM kernels). If features are on very different scales (e.g.
fAlpha ranges roughly 0-90, fAsym ranges into the hundreds), the
larger-scale feature would dominate any distance calculation, making the
smaller-scale features almost meaningless. Normalization fixes this by
rescaling every feature to have mean 0 and standard deviation 1
(Lecture 3b, Section 7)."""

mean_train = X_train.mean()
std_train = X_train.std()

X_train_norm = (X_train - mean_train) / std_train
X_test_norm = (X_test - mean_train) / std_train

"""CRITICAL TRAP (Lecture 3b, Section 7 and Lecture 7):
The mean and standard deviation used above are computed from X_train
ONLY. We then reuse those exact same values to normalize X_test. We never
compute statistics on the combined train+test data, and we never compute
separate statistics for the test set. Doing so would let information about
the test set leak into preprocessing, making the later evaluation
dishonest — even though the test labels themselves are never touched."""

print(f"\n--- Normalization ---")
print("Training set mean after normalization (should be ~0 for every feature):")
print(X_train_norm.mean().round(3))

print("\nTraining set std after normalization (should be ~1 for every feature):")
print(X_train_norm.std().round(3))

print("\nTest set mean after normalization (will NOT be exactly 0 — this is expected):")
print(X_test_norm.mean().round(3))

"""WHAT TO LOOK FOR:
The training set's mean and std should land almost exactly at 0 and 1,
since we computed the normalization directly from this data.
The test set's mean and std will be CLOSE to 0 and 1 but not exact — this
is expected and correct. If the test set's values came out exactly at 0
and 1, that would actually be a red flag, suggesting normalization was
computed on the test set itself rather than reused from training."""


# =============================================================================
# SECTION 6: SAVE THE PREPARED DATA
# =============================================================================
"""We save the split and normalized data to disk so that 03_baseline_knn.py,
04_svm.py, and 05_random_forest.py can all load the exact same prepared
data, rather than each repeating this preprocessing independently. This
also guarantees every model is evaluated on an identical train/test split,
making later comparisons in 06_evaluation.py fair."""

os.makedirs(os.path.join("data", "processed"), exist_ok=True)

X_train_norm.to_csv(os.path.join("data", "processed", "X_train.csv"), index=False)
X_test_norm.to_csv(os.path.join("data", "processed", "X_test.csv"), index=False)
y_train.to_csv(os.path.join("data", "processed", "y_train.csv"), index=False)
y_test.to_csv(os.path.join("data", "processed", "y_test.csv"), index=False)

print(f"\nProcessed data saved to: {os.path.join('data', 'processed')}")
print("Files: X_train.csv, X_test.csv, y_train.csv, y_test.csv")