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
meaning no information from the test set leaks into training"""

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

# =============================================================================
# SECTION 2: MISSING VALUES (NO ACTION NEEDED) & CHECK FOR DUPLICATE ROWS (NO REMOVAL)
# =============================================================================
"""01_eda.py confirmed 0 missing values across every column, and 0 hidden
missing-value placeholders (e.g. "?", "NA"). Lecture 10's data preparation
checklist includes handling missing values (fill or delete) as a required
step — we explicitly note here that this step requires no action for this
dataset, rather than silently skipping it."""

print(f"\n--- Missing values check (carried over from 01_eda.py) ---")
print("Confirmed: 0 missing values, 0 hidden missing-value placeholders.")
print("Decision: no filling in or row deletion needed.")

"""01_eda.py already checked for duplicate rows. We check again here just
to document the count here as well before splitting.

WHY NOT JUST REMOVE THEM?
Removing a data point is only
acceptable if you are certain it's an error AND can explain why. "My
results look better without it" is never a valid reason. We have no proof
that duplicate rows here are logging errors rather than genuinely repeated
or very similar physical events from the simulation that generated this
dataset. So we keep them, and only revisit this decision if we later find
concrete evidence they are mistakes."""

"""duplicate rows are far more plausible to be real, independent simulated events 
that happen to land on very similar parameter values — not logging errors."""

n_duplicates = df.duplicated().sum()
print(f"\n--- Duplicate rows (kept, not removed) ---")
print(f"Number of duplicate rows: {n_duplicates}")
print("Decision: keeping all rows - no evidence these are data errors yet")

# =============================================================================
# SECTION 3: TRAIN / TEST SPLIT
# =============================================================================
"""We split the data into a training set (80%) and a test set (20%).

WHY 80/20?
With 19020 samples, an 80/20 split still leaves a large test set (~3800
samples) — large enough to give a stable, reliable estimate of performance,
while keeping most of the data available for training.

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
Both the training set and the test set should show around the same
class proportions as the full dataset (~65% gamma, ~35% hadron). If the
proportions differ by a lot, the split did not stratify correctly."""

"""IMPORTANT RULE GOING FORWARD (Lecture 10 — "data snooping"):
From this point on, X_test and y_test are LOCKED AWAY. We do not look at
them again, plot them, or use them to make any decision until the very
final evaluation in 06_evaluation.py. Any hyperparameter tuning in
03/04/05 will use cross-validation on X_train/y_train only."""


# =============================================================================
# SECTION 4: NORMALIZE FEATURES
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
separate statistics for the test set. If we computed the test set's
own statistics, we would be using information from the test set to decide
how the data is transformed — even before any model sees it. That's a
form of "peeking" at the test set. It would make the test set no longer
a fair, untouched measure of real-world performance, since the
preprocessing itself was shaped by data the model is supposed to have
never seen."""

print(f"\n--- Normalization ---")
print("Training set mean after normalization:")
print(X_train_norm.mean().round(3))

print("\nTraining set std after normalization:")
print(X_train_norm.std().round(3))

print("\nTest set mean after normalization:")
print(X_test_norm.mean().round(3))

print("\nTest set std after normalization (will NOT be exactly 1 — this is expected):")
print(X_test_norm.std().round(3))

"""WHY EXACTLY 0 AND 1 FOR THE TRAINING SET?
This is NOT because normalization makes the data normally distributed —
it does not change the shape of a feature's distribution at all (a skewed
feature stays skewed). It is pure arithmetic: subtracting the mean from
every value always shifts the average to exactly 0, and dividing every
value by the standard deviation always rescales the spread to exactly 1.
Since mean_train/std_train were computed directly FROM the training data,
applying this formula to that same data gives EXACTLY 0/1 (any tiny
deviation shown is only floating-point rounding, not a real difference).

WHY NOT EXACTLY 0/1 FOR THE TEST SET?
We reused the TRAINING set's mean/std on the test set, not the test set's
own. The test set is a different random sample with its own slightly
different actual mean/std, so after applying the training-derived
formula, its mean/std will be CLOSE to 0/1 but not exact — this is
expected and correct. If the test set's mean/std came out exactly 0/1,
that would mean we had instead computed its own statistics and normalized
with those — exactly the leak we are trying to avoid.

RANDOM FOREST:
Decision trees split one feature at a time using a threshold (e.g.
"fAlpha < 17?"), not a distance calculation across all features at once.
This means normalization has no real effect on Random Forest's splits or
performance. We still apply the same normalized data to it for pipeline
consistency across all three models, but we will note in 05_random_forest.py
that this step was not strictly necessary for that specific model."""


# =============================================================================
# SECTION 5: SAVE THE PREPARED DATA
# =============================================================================
"""We save the split and normalized data to disk so that 03_baseline_knn.py,
04_svm.py, and 05_random_forest.py can all load the exact same prepared
data, rather than each repeating this preprocessing independently. This
also guarantees every model is evaluated on an identical train/test split,
making later comparisons in 06_evaluation.py fair."""

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

"""
Why normalize — and do your specific models need it?

- kNN: needs it. Lecture 3b is explicit — kNN relies on distance, and if one 
feature ranges 0–10000 and another 0–1, the larger one dominates the distance 
calculation entirely, making the smaller feature meaningless. You have exactly this 
kind of scale mismatch (fAlpha ~0–90, fAsym into the hundreds).

- SVM: needs it. Lecture 9's hyperplane math and especially the RBF 
kernel (exp(-γ·distance²)) are distance-based the same way kNN is — unscaled 
features would distort the margin and kernel calculations.

- Random Forest: does NOT need it. Decision trees (Lecture 9) split by asking 
"is feature X < threshold?" one feature at a time — there's no distance 
calculation and no comparison across features happening simultaneously. 
A tree splitting on fAlpha at value 17 works identically whether fAlpha is 
in its raw units or normalized, since the split point and resulting groups 
are the same either way.

So normalization is necessary for your base classifier and one of your advanced ones, 
but technically unnecessary for Random Forest. Worth stating this explicitly 
somewhere — it's exactly the kind of "why did you do this" reasoning the professor 
grades. You can still apply the same normalized data to Random Forest (it won't hurt 
accuracy), but you should be ready to say "we know RF doesn't strictly need this, we 
used the same prepared data for pipeline consistency across models."
"""

"""
This reasoning of why these 3 models where chosen over others belongs in your 
presentation (the "Method" section, per the Scientific Presentations slide deck:
— "Explain how you constructed the machine learning evaluation. This includes: 
Model and feature selection...").

A reasonable justification structure, grounded in your own course material:

- kNN (base): simple, parameter-free, easy to explain, the natural starting baseline 
(Lecture 3a explicitly frames it as the "first classifier that actually works").

- SVM (advanced 1): continuous, fairly clean features (Hillas parameters) — 
exactly the kind of data your notes say SVM with RBF handles well; also produces 
ROC curves naturally, which the dataset explicitly requires (Lecture 9, Section 20).

- Random Forest (advanced 2): handles the class imbalance more gracefully than 
purely distance-based methods, gives feature importance for free (useful given 
your fAlpha hypothesis from EDA), and is comparatively robust to scale/outliers — a 
deliberately different "type" of classifier from kNN/SVM, which is valuable since 
your project plan even recommends picking models that make different kinds of errors.

- Why not LDA/QDA/Naive Bayes/logistic regression? You could mention you considered 
them, but they all make a distributional assumption (Gaussian, or feature independence) 
that you have no strong evidence holds for this Hillas-parameter data (and your 
EDA showed several features with non-symmetric, skewed distributions) — whereas 
kNN, SVM-RBF, and Random Forest make no such assumption.

"""