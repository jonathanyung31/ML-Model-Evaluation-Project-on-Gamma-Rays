# =============================================================================
# 02_preprocessing.py — Data Preparation
# =============================================================================

# =============================================================================
# SECTION 1: LOAD THE DATA
# =============================================================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

column_names = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1",
    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"
]

df = pd.read_csv(
    os.path.join("data", "magic04.data"),
    header=None,
    names=column_names
)

print(f"Shape before cleaning: {df.shape}")

# Since we only read from magic04.data and never write back to it, the
# original file is never at risk — but we make this explicit by working
# with an in-memory copy throughout.
df = df.copy()

df = df.drop_duplicates(keep="first").reset_index(drop=True)

# SECTION 2: MISSING VALUES (NO ACTION NEEDED) & CHECK FOR DUPLICATE ROWS (NO REMOVAL)

print(f"\n--- Missing values check (carried over from 01_eda.py) ---")
print("Confirmed: 0 missing values, 0 hidden missing-value placeholders.")
print("Decision: no filling in or row deletion needed.")

n_duplicates = df.duplicated().sum()
print(f"\n--- Duplicate rows (kept, not removed) ---")
print(f"Number of duplicate rows: {n_duplicates}")
print("Decision: keeping all rows - no evidence these are data errors yet")


# SECTION 3: TRAIN / TEST SPLIT

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


# SECTION 4: NORMALIZE FEATURES


mean_train = X_train.mean()
std_train = X_train.std()

X_train_norm = (X_train - mean_train) / std_train
X_test_norm = (X_test - mean_train) / std_train



print(f"\n--- Normalization ---")
print("Training set mean after normalization:")
print(X_train_norm.mean().round(3))

print("\nTraining set std after normalization:")
print(X_train_norm.std().round(3))

print("\nTest set mean after normalization:")
print(X_test_norm.mean().round(3))

print("\nTest set std after normalization (will NOT be exactly 1 — this is expected):")
print(X_test_norm.std().round(3))




# SECTION 5: SAVE THE PREPARED DATA


# create folder if it doesn't already exist
os.makedirs(os.path.join("data", "processed"), exist_ok=True)   

# Save the normalized training features to a CSV file
X_train_norm.to_csv(os.path.join("data", "processed", "X_train.csv"), index=False)

# Save the normalized test features to a CSV file
X_test_norm.to_csv(os.path.join("data", "processed", "X_test.csv"), index=False)

# Save the training labels to a CSV file
y_train.to_csv(os.path.join("data", "processed", "y_train.csv"), index=False)

# Save the test labels to a CSV file
y_test.to_csv(os.path.join("data", "processed", "y_test.csv"), index=False)

print(f"\nProcessed data saved to: {os.path.join('data', 'processed')}")
print("Files: X_train.csv, X_test.csv, y_train.csv, y_test.csv")

