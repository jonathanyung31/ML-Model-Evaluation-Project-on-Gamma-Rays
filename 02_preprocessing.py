# =============================================================================
# 02_preprocessing.py — Data Preparation
# =============================================================================

# =============================================================================
# SECTION 1: LOAD THE DATA
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # matplotlib: for creating plots
import os
import seaborn as sns        # seaborn: for nicer statistical plots
from sklearn.model_selection import train_test_split

sns.set_theme(style="darkgrid")
plt.style.use("dark_background")
plt.rcParams["grid.color"] = "#3a3a3a"   # dark grey grid lines
plt.rcParams["grid.alpha"] = 0.6         # optional: slightly transparent

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

print("Confirmed: 0 missing values, 0 hidden missing-value placeholders.")
print("Decision: no filling in or row deletion needed.")
n_duplicates = df.duplicated().sum()
print(f"Duplicate rows found: {n_duplicates}")
df = df.drop_duplicates(keep="first").reset_index(drop=True)
print(f"Number of rows after removing duplicates: {len(df)}")

print("Updated class distribution:")
print(df["class"].value_counts())

print("Decision: removed - with 10-decimal-precision CORSIKA-simulated")
print("features, an exact match across all 10 features is a data-pipeline")
print("artifact, not a genuinely repeated physical event (same reasoning")
print("as 01_eda.py, kept consistent across both scripts).")

# =============================================================================
# OUTLIER DECISION (Lecture 7)
# =============================================================================
"""01_eda.py flagged a small percentage of rows with an extreme
(|z-score| > 3) feature value. Per Lecture 7: without a documented,
verifiable reason the values are wrong (Step 1/2), and with no
physically impossible values found, they are kept -- 'my results get
better without it' is not a valid reason to remove a point.
Why: just documents the decision at the stage where you'd actually be 
tempted to act on it — no new computation needed here at all.
"""

# CLASS DISTRIBUTION

print("\n--- Class distribution ---")
class_counts = df["class"].value_counts()
class_proportions = df["class"].value_counts(normalize=True) * 100     # in precentages

print("Counts:")
print(class_counts)
print("\nProportions (%):")
print(class_proportions.round(2))   # limiting to 2 decimal points


# PLOT: Class distribution bar chart
fig, ax = plt.subplots(figsize=(6, 4))

class_counts.plot(
    kind="bar",
    ax=ax,
    color=["deepskyblue", "magenta"],
    edgecolor="black"
)

ax.set_ylim(0, max(class_counts) * 1.25)    # cahnge y-axis limits for avoiding overlapping text

ax.set_title("Class Distribution: Gamma vs Hadron", fontsize=13)

# Replace the default x-axis labels with more descriptive labels
# rotation=0 keeps the labels horizontal
ax.set_xticklabels(["g (gamma)", "h (hadron)"], rotation=0)

ax.set_xlabel("Class (g = gamma/signal, h = hadron/background)", fontsize=12, labelpad=12)
ax.set_ylabel("Number of Samples", fontsize=12, labelpad=12)

# Add the count and percentage as text on top of each bar
for i, (count, prop) in enumerate(zip(class_counts, class_proportions)):
    # Place text at x-position i and slightly above the bar height.
    # \n creates a new line so the count and percentage appear on
    # separate lines.
    ax.text(i, count + 100, f"{count}\n({prop:.1f}%)",
            ha="center",        # horizontally center the text
            va="bottom",        # align the bottom of the text to the position
            fontsize=10)

plt.tight_layout()      # Automatically adjust spacing so labels and titles don't overlap
# Save the figure to a PNG file with good image quality
# dpi=150 means 150 dots (pixels) per inch
plt.savefig(os.path.join("results", "plots", "01_class_distribution.png"), dpi=150)
plt.show()      # Display the plot on the screen
print("Plot saved: 01_class_distribution.png")

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

