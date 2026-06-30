# =============================================================================
# 03_baseline_knn.py — Base Experiment: k-Nearest Neighbors
# =============================================================================
"""WHAT IS THIS FILE?
This is the third step of the project: training our baseline classifier,
kNN. kNN is our base model because it is simple, parameter-free (no
learned weights), and easy to explain (Lecture 3a) — it gives us a
starting point that any "advanced" classifier (SVM, Random Forest) needs
to beat to justify its added complexity.

WHY DOES KNN NEED NORMALIZED DATA?
kNN classifies a point by finding its closest neighbors using distance
(Lecture 3a). If features are on different scales, the largest-scale
feature would dominate every distance calculation. This is why we use the
already-normalized data produced by 02_preprocessing.py, not the raw data."""

# =============================================================================
# SECTION 1: IMPORTS AND LOAD PREPROCESSED DATA
# =============================================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
# Import tools for cross-validation and stratified k-fold splitting
from sklearn.model_selection import cross_val_score, StratifiedKFold    
# Import functions for evaluating classification models
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    ConfusionMatrixDisplay
)

X_train = pd.read_csv(os.path.join("data", "processed", "X_train.csv"))
X_test = pd.read_csv(os.path.join("data", "processed", "X_test.csv"))
y_train = pd.read_csv(os.path.join("data", "processed", "y_train.csv")).squeeze()   # into pandas series
y_test = pd.read_csv(os.path.join("data", "processed", "y_test.csv")).squeeze()      # into pandas series

print("Loaded preprocessed data:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

"""NOTE: .squeeze() converts a single-column DataFrame (what to_csv/read_csv
gives us back for y_train/y_test) into a plain pandas Series, which is the
format sklearn expects for labels."""


# =============================================================================
# SECTION 2: CHOOSE k VALUES TO TEST (HYPERPARAMETER TUNING)
# =============================================================================
"""k is kNN's only real hyperparameter (Lecture 3a/3b) — it is NOT learned
from the data, we have to choose it ourselves. We test several values and
use cross-validation on the TRAINING set only to pick the best one,
exactly as Lecture 3b/7 describes: the test set must not be touched for
any decision-making, only for the final reported result.

WHY THESE SPECIFIC VALUES?
We try a spread from very small (k=1, most flexible, most overfitting
risk) to larger (k=25, smoother, more underfitting risk), matching the
overfitting-vs-underfitting tradeoff shown in Lecture 3b, Section 1-2."""

k_values = [1, 3, 5, 10, 15, 25]

"""WHY CROSS-VALIDATION INSTEAD OF A SINGLE VALIDATION SPLIT?
Lecture 7 explains that with a single random split, the result can be
unstable — depending on which points happen to land in the validation
set. k-fold cross-validation (we use 5 folds here, a typical choice per
Lecture 7) averages over several splits, giving a more reliable estimate
of how each k value will generalize."""

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

"""WHY STRATIFIED K-FOLD SPECIFICALLY?
Just like our train/test split, each fold must also preserve the
~65/35 class ratio - otherwise some folds could end up with a very
different class balance than others, making the cross-validation
results noisier than necessary."""

cv_errors = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)

    """We use the classwise average classification rate (balanced
    accuracy) here, NOT plain accuracy, since the dataset documentation
    and Lecture 3b both warn against accuracy on imbalanced classes."""
    scores = cross_val_score(
        knn, X_train, y_train,
        cv=cv_strategy,
        scoring="balanced_accuracy"
    )
    mean_error = 1 - scores.mean()   # convert accuracy to an error rate
    cv_errors.append(mean_error)
    print(f"k={k}: mean CV balanced-accuracy error = {mean_error:.3f}")


# =============================================================================
# SECTION 3: PLOT VALIDATION ERROR vs k
# =============================================================================
"""This plot directly recreates the kind of overfitting/underfitting curve
shown in Lecture 3b, Section 1 (kNN with simulated data): small k overfits
(jagged, memorizes training data), large k underfits (too smooth). We
expect to see a U-shape or similar — error high at very small k, dropping
to a minimum, then rising again at very large k."""

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(k_values, cv_errors, marker="o", color="steelblue")
ax.set_xlabel("k (number of neighbors)", fontsize=11)
ax.set_ylabel("Cross-validated error (1 - balanced accuracy)", fontsize=11)
ax.set_title("kNN: Validation Error vs k", fontsize=13)
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "05_knn_k_selection.png"), dpi=150)
plt.show()
print("Plot saved: 05_knn_k_selection.png")

best_k = k_values[np.argmin(cv_errors)]
print(f"\nBest k selected via cross-validation: {best_k}")

"""WHAT TO LOOK FOR:
The k with the lowest cross-validated error is our chosen hyperparameter.
This decision was made using ONLY the training set (via cross-validation)
— the test set has not been touched yet."""


# =============================================================================
# SECTION 4: TRAIN FINAL MODEL WITH BEST k, EVALUATE ON TEST SET
# =============================================================================
"""Now, and only now, do we touch the test set. This is the one and only
evaluation it gets (Lecture 10 — "data snooping" warning): we train the
final model (with the k chosen above) on the full training set, and
report its performance on the test set as our honest, final result."""

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)

y_pred = final_knn.predict(X_test)
y_pred_proba = final_knn.predict_proba(X_test)[:, list(final_knn.classes_).index("g")]

"""WHY predict_proba AND NOT JUST predict?
predict() gives a hard class label (g or h). But to compute ROC curves
and auROC (Lecture 3b/5b), we need a continuous probability/score per
point, so we can sweep a threshold across all possible values, exactly
as Lecture 5b's ROC curve section describes."""


# =============================================================================
# SECTION 5: CONFUSION MATRIX (ABSOLUTE AND RELATIVE)
# =============================================================================
"""Lecture 3b explicitly recommends showing BOTH the absolute confusion
matrix (raw counts) and the relative confusion matrix (each row divided
by its true class total) — the relative version makes the per-class
classification rate easy to read directly, which matters here since our
classes are imbalanced."""

cm = confusion_matrix(y_test, y_pred, labels=["g", "h"])
cm_relative = cm.astype(float) / cm.sum(axis=1, keepdims=True)

print("\n--- Confusion Matrix (absolute) ---")
print(cm)

print("\n--- Confusion Matrix (relative, per true class) ---")
print(np.round(cm_relative, 3))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ConfusionMatrixDisplay(cm, display_labels=["g", "h"]).plot(ax=axes[0], cmap="Blues")
axes[0].set_title("Absolute Confusion Matrix")

ConfusionMatrixDisplay(cm_relative, display_labels=["g", "h"]).plot(ax=axes[1], cmap="Blues")
axes[1].set_title("Relative Confusion Matrix")

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "06_knn_confusion_matrix.png"), dpi=150)
plt.show()
print("Plot saved: 06_knn_confusion_matrix.png")


# =============================================================================
# SECTION 6: SENSITIVITY, SPECIFICITY, auROC
# =============================================================================
"""WHY NOT JUST ACCURACY?
Both the dataset documentation and Lecture 3b are explicit: with ~65/35
imbalanced classes, accuracy can look good while the model is still bad
at catching the minority class. We compute sensitivity and specificity
directly from the confusion matrix (Lecture 3b, Section 10), and auROC
as our single most meaningful summary number (Lecture 3b/5b)."""

tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)   # true positive rate, here "h" is the second label - adjust below
specificity = tn / (tn + fp)

"""NOTE ON LABEL ORDER:
confusion_matrix(labels=["g","h"]) puts "g" first. So in our cm, position
[0,0]=g predicted as g, etc. We ravel() assuming a 2-class layout, but to
be precise about which class is "positive," we explicitly recompute below
using fAlpha's class wisdom: gamma (g) is our signal/positive class,
matching the dataset's own framing (g = signal, h = background)."""

tp = cm[0, 0]   # true g predicted as g
fn = cm[0, 1]   # true g predicted as h (missed signal)
fp = cm[1, 0]   # true h predicted as g (false alarm)
tn = cm[1, 1]   # true h predicted as h

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"\nSensitivity (gamma correctly identified): {sensitivity:.3f}")
print(f"Specificity (hadron correctly identified): {specificity:.3f}")

auroc = roc_auc_score((y_test == "g").astype(int), y_pred_proba)
print(f"auROC: {auroc:.3f}")

"""WHAT TO LOOK FOR:
auROC close to 1.0 = strong separation between classes (Lecture 3b/5b).
auROC close to 0.5 = no better than random guessing.
We expect a reasonably high auROC here given fAlpha's strong individual
separation found in 01_eda.py."""


# =============================================================================
# SECTION 7: ROC CURVE AND REPORTING AT REQUIRED FPR THRESHOLDS
# =============================================================================
"""The dataset documentation (magic04.names) explicitly specifies the FPR
thresholds at which sensitivity should be compared between classifiers:
0.01, 0.02, 0.05, 0.1, 0.2. We compute the ROC curve and report
sensitivity at exactly these points, so this baseline can be directly
compared against SVM and Random Forest later in 06_evaluation.py."""

fpr, tpr, thresholds = roc_curve((y_test == "g").astype(int), y_pred_proba)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color="steelblue", label=f"kNN (auROC = {auroc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guessing")
ax.set_xlabel("False Positive Rate (1 - specificity)", fontsize=11)
ax.set_ylabel("True Positive Rate (sensitivity)", fontsize=11)
ax.set_title("kNN ROC Curve", fontsize=13)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "07_knn_roc_curve.png"), dpi=150)
plt.show()
print("Plot saved: 07_knn_roc_curve.png")

required_fpr_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]

print("\n--- Sensitivity at required FPR thresholds (per dataset documentation) ---")
for target_fpr in required_fpr_thresholds:
    idx = np.argmin(np.abs(fpr - target_fpr))
    print(f"FPR ~{target_fpr}: sensitivity = {tpr[idx]:.3f} (actual FPR at this point: {fpr[idx]:.3f})")

"""WHAT TO LOOK FOR:
This table is the exact comparison structure the dataset's original
authors intended (magic04.names, Section 9). We will reuse this same
table format for SVM and Random Forest, so all three models can be lined
up side-by-side in 06_evaluation.py."""


# =============================================================================
# SECTION 8: NOTE ON FEATURE SELECTION AND HIGH LEVERAGE / OUTLIERS
# =============================================================================
"""Per Lecture 3b (mistake #2) and Lecture 7, feature selection and any
outlier/high-leverage analysis must be model-specific, not done once in
preprocessing. For this baseline kNN model, we deliberately use all 10
features without selection, to establish a true baseline before any
feature engineering. Outlier/high-leverage analysis specific to kNN's
predictions (Lecture 7: misclassified points farthest from the decision
boundary) is left for a deeper pass if time allows, but is not required
for the baseline to be valid."""