# =============================================================================
# 05_random_forest.py — Advanced Classifier 2: Random Forest
# =============================================================================
"""WHAT IS THIS FILE?
This is the fifth step of the project: training our second advanced
classifier, a Random Forest. Random Forest is "advanced" relative to our
kNN baseline because it combines many decision trees (Lecture 9) through
bagging and random feature selection, producing a much more stable and
accurate classifier than any single tree alone.

WHY RANDOM FOREST FOR THIS DATASET?
Three reasons, all grounded in what we found in 01_eda.py:
1. Several features (fAsym, fM3Long, fM3Trans) showed weak individual
   separation in our histograms — Random Forest can still find value in
   weak features by combining them across many trees, whereas a linear
   model would underweight them.
2. The ~65/35 class imbalance is handled more gracefully by Random Forest
   than by distance-based methods like kNN, since tree splits are based
   on information gain (Gini/RSS), not distances.
3. Random Forest gives feature importance for free (Lecture 9, Section
   39) — this lets us directly check whether fAlpha, which our EDA
   identified as the most discriminative feature, truly dominates when
   a real classifier is trained. This closes the loop between our EDA
   hypotheses and our modeling results.

DOES RANDOM FOREST NEED NORMALIZED DATA?
No — decision trees split one feature at a time using a threshold
(e.g. "fAlpha < 17?"), not a distance calculation across all features
simultaneously (Lecture 9, Section 27). Normalization has no effect on
where splits land or how the tree grows. We still use the normalized
data from 02_preprocessing.py for pipeline consistency across all three
models, but this is not a requirement for Random Forest specifically."""


# =============================================================================
# SECTION 1: IMPORTS AND LOAD PREPROCESSED DATA
# =============================================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    ConfusionMatrixDisplay
)

X_train = pd.read_csv(os.path.join("data", "processed", "X_train.csv"))
X_test = pd.read_csv(os.path.join("data", "processed", "X_test.csv"))
y_train = pd.read_csv(os.path.join("data", "processed", "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join("data", "processed", "y_test.csv")).squeeze()

feature_names = X_train.columns.tolist()
# store feature names now, before any numpy conversion, so we can label
# the feature importance plot in Section 7 correctly

print("Loaded preprocessed data:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")


# =============================================================================
# SECTION 2: HYPERPARAMETERS OF RANDOM FOREST
# =============================================================================
"""Random Forest has four main hyperparameters (Lecture 9, Section 38):

1. n_estimators (B): number of trees in the forest. More trees = more
   stable predictions, but diminishing returns after a point. The error
   rate stabilizes once you have enough trees — adding more after that
   does not hurt but also does not help (Lecture 9, Section 37).

2. max_features (m): number of features randomly considered at each
   split. This is the key difference from plain bagging — by only
   considering a random subset of features at each split, the trees are
   forced to be different from each other (decorrelated), which reduces
   variance when averaged. Typical default: sqrt(p) where p = number of
   features (Lecture 9, Section 38). With p=10 features, sqrt(10) ≈ 3.

3. max_depth: how deep each tree is allowed to grow. Deeper trees =
   more flexible, higher overfitting risk (Lecture 3b, Section 2).
   None means trees grow until all leaves are pure or have fewer than
   min_samples_split samples.

4. The split criterion: Gini index or entropy (Lecture 9, Section 32).
   Both measure node impurity — lower impurity = better split. Gini is
   the default and generally performs similarly to entropy in practice."""


# =============================================================================
# SECTION 3: HYPERPARAMETER TUNING VIA CROSS-VALIDATION
# =============================================================================
"""We tune n_estimators and max_depth using cross-validation on the
training set only (Lecture 3b/7) — the test set remains locked.
max_features is left at its default (sqrt(p) ≈ 3), which Lecture 9
identifies as the standard recommended starting value.

WHY THESE SPECIFIC RANGES?
n_estimators: we test [50, 100, 200, 300] — small enough to run quickly,
large enough to see where the error stabilizes. Lecture 9 notes that
beyond a certain number of trees, adding more changes results very little.
max_depth: we test [None, 5, 10, 20] — None (fully grown trees) is the
bagging default; shallower depths add regularization (Lecture 3b) and
may generalize better if full trees overfit."""

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

"""WHY STRATIFIED K-FOLD AGAIN?
Same reasoning as 03_baseline_knn.py: each fold must preserve the
~65/35 class ratio so cross-validation results are not distorted by
folds with an unrepresentative class balance."""

n_estimators_values = [50, 100, 200, 300]
max_depth_values = [None, 5, 10, 20]

best_score = -np.inf
best_params = {}
results = []

print("\n--- Hyperparameter search (n_estimators x max_depth) ---")

for n_est in n_estimators_values:
    for depth in max_depth_values:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            max_features="sqrt",    # sqrt(p) — Lecture 9 default
            random_state=42,
            n_jobs=-1               # use all available CPU cores
        )
        scores = cross_val_score(
            rf, X_train, y_train,
            cv=cv_strategy,
            scoring="balanced_accuracy"
        )
        mean_score = scores.mean()
        std_score = scores.std()
        mean_error = 1 - mean_score

        results.append({
            "n_estimators": n_est,
            "max_depth": depth,
            "mean_balanced_acc": mean_score,
            "std": std_score,
            "mean_error": mean_error
        })

        print(f"n_est={n_est}, depth={depth}: "
              f"balanced_acc={mean_score:.3f} (std={std_score:.3f})")

        if mean_score > best_score:
            best_score = mean_score
            best_params = {"n_estimators": n_est, "max_depth": depth}

print(f"\nBest parameters: {best_params}")
print(f"Best CV balanced accuracy: {best_score:.3f}")

"""WHY BALANCED ACCURACY FOR TUNING?
Same justification as 03_baseline_knn.py: it directly addresses the
class imbalance found in 01_eda.py. Final headline comparison between
models still uses auROC — balanced accuracy here is only for selecting
hyperparameters during cross-validation, not the final reported metric.
The std across folds is also printed to check stability of the estimate,
connecting to Lecture 7's point that a single number without spread
information can be misleading."""


# =============================================================================
# SECTION 4: VISUALIZE HYPERPARAMETER SEARCH RESULTS
# =============================================================================
"""We show a heatmap of balanced accuracy across all (n_estimators,
max_depth) combinations, exactly as we did for SVM's (C, gamma) grid in
04_svm.py. This makes the best region visually obvious and lets us check
whether the best combination sits at an edge of the grid (which would
suggest we should extend the search range)."""

results_df = pd.DataFrame(results)

pivot = results_df.pivot(
    index="max_depth",
    columns="n_estimators",
    values="mean_balanced_acc"
)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([str(d) for d in pivot.index])
ax.set_xlabel("n_estimators (number of trees)", fontsize=11)
ax.set_ylabel("max_depth", fontsize=11)
ax.set_title("Random Forest: Balanced Accuracy (CV)", fontsize=13)
fig.colorbar(im, ax=ax, label="Balanced Accuracy")

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax.text(j, i, f"{pivot.values[i, j]:.3f}",
                ha="center", va="center", color="white", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "11_rf_hyperparameter_search.png"), dpi=150)
plt.show()
print("Plot saved: 11_rf_hyperparameter_search.png")

"""WHAT TO LOOK FOR:
The best (n_estimators, max_depth) should not sit at an edge of the
grid — if n_estimators=300 is still clearly best, the error may not
have stabilized yet and we should try larger values. If max_depth=None
(fully grown) is clearly best, then pruning hurts this dataset and
trees should be left to grow fully."""


# =============================================================================
# SECTION 5: TRAIN FINAL MODEL ON FULL TRAINING SET, EVALUATE ON TEST SET
# =============================================================================
"""Same principle as 04_svm.py Section 5: retrain on the FULL training
set (not any subsample) with the best hyperparameters found above, then
touch the test set exactly once for the honest, final evaluation.
(Lecture 10 Step 6: "use as much data as you can at this stage.")"""

final_rf = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
final_rf.fit(X_train, y_train)

y_pred = final_rf.predict(X_test)
y_pred_proba = final_rf.predict_proba(X_test)[:, list(final_rf.classes_).index("g")]

"""WHY predict_proba?
Same reasoning as 03_baseline_knn.py and 04_svm.py: we need a continuous
score per point to build the ROC curve and compute auROC, rather than
just a hard class label (Lecture 3b/5b)."""


# =============================================================================
# SECTION 6: CONFUSION MATRIX (ABSOLUTE AND RELATIVE)
# =============================================================================
"""Same structure as 03_baseline_knn.py and 04_svm.py for direct
comparison in 06_evaluation.py."""

cm = confusion_matrix(y_test, y_pred, labels=["g", "h"])
cm_relative = cm.astype(float) / cm.sum(axis=1, keepdims=True)

print("\n--- Confusion Matrix (absolute) ---")
print(cm)

print("\n--- Confusion Matrix (relative, per true class) ---")
print(np.round(cm_relative, 3))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ConfusionMatrixDisplay(cm, display_labels=["g", "h"]).plot(ax=axes[0], cmap="Oranges")
axes[0].set_title("Absolute Confusion Matrix")

ConfusionMatrixDisplay(cm_relative, display_labels=["g", "h"]).plot(ax=axes[1], cmap="Oranges")
axes[1].set_title("Relative Confusion Matrix")

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "12_rf_confusion_matrix.png"), dpi=150)
plt.show()
print("Plot saved: 12_rf_confusion_matrix.png")


# =============================================================================
# SECTION 7: FEATURE IMPORTANCE
# =============================================================================
"""One of Random Forest's key advantages over kNN and SVM is that it
gives feature importance for free (Lecture 9, Section 39): for each
feature, the total reduction in Gini impurity caused by splits on that
feature is summed across all trees and all nodes. A larger total means
that feature was consistently useful for making good splits.

WHY IS THIS IMPORTANT FOR OUR PROJECT?
In 01_eda.py we hypothesized from histograms and scatter plots that
fAlpha is the most discriminative feature (gamma showers point toward
the source, giving low fAlpha values). Feature importance from a trained
classifier is a much stronger, model-backed confirmation or correction
of that hypothesis — it closes the loop between our EDA findings and
our modeling results, which is exactly the kind of reasoning the
professor grades."""

importances = final_rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]   # sort descending

print("\n--- Feature Importance (sorted) ---")
for i in sorted_idx:
    print(f"{feature_names[i]}: {importances[i]:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    range(len(importances)),
    importances[sorted_idx],
    color="darkorange",
    edgecolor="black"
)
ax.set_xticks(range(len(importances)))
ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45, ha="right")
ax.set_ylabel("Feature Importance (mean Gini reduction)", fontsize=11)
ax.set_title("Random Forest: Feature Importance", fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "13_rf_feature_importance.png"), dpi=150)
plt.show()
print("Plot saved: 13_rf_feature_importance.png")

"""WHAT TO LOOK FOR:
We expect fAlpha to rank at or near the top, confirming our EDA
hypothesis. Features that ranked as weak discriminators in the
histograms (fAsym, fM3Long, fM3Trans) may still show meaningful
importance here — Random Forest can exploit weak features in combination
across many trees, even when they appear weak individually. Any
surprising ranking relative to the EDA findings is worth discussing
explicitly in 06_evaluation.py."""


# =============================================================================
# SECTION 8: SENSITIVITY, SPECIFICITY, auROC
# =============================================================================
"""Same label convention as 03_baseline_knn.py and 04_svm.py:
gamma (g) is the signal/positive class."""

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


# =============================================================================
# SECTION 9: ROC CURVE AND REPORTING AT REQUIRED FPR THRESHOLDS
# =============================================================================
"""Same structure as 03_baseline_knn.py and 04_svm.py, using the same
FPR thresholds from magic04.names (0.01, 0.02, 0.05, 0.1, 0.2) so all
three models line up directly in 06_evaluation.py."""

fpr, tpr, thresholds = roc_curve((y_test == "g").astype(int), y_pred_proba)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color="darkorange", label=f"Random Forest (auROC = {auroc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guessing")
ax.set_xlabel("False Positive Rate (1 - specificity)", fontsize=11)
ax.set_ylabel("True Positive Rate (sensitivity)", fontsize=11)
ax.set_title("Random Forest ROC Curve", fontsize=13)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "14_rf_roc_curve.png"), dpi=150)
plt.show()
print("Plot saved: 14_rf_roc_curve.png")

required_fpr_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]

print("\n--- Sensitivity at required FPR thresholds (per dataset documentation) ---")
for target_fpr in required_fpr_thresholds:
    idx = np.argmin(np.abs(fpr - target_fpr))
    print(f"FPR ~{target_fpr} (closest available point: {fpr[idx]:.3f}): "
          f"sensitivity = {tpr[idx]:.3f}")


# =============================================================================
# SECTION 10: NOTE ON FEATURE SELECTION AND HIGH LEVERAGE / OUTLIERS
# =============================================================================
"""Per Lecture 3b (mistake #2) and Lecture 7, feature selection and
outlier/high-leverage analysis are model-specific. For this Random Forest
model we use all 10 features, consistent with our kNN baseline and SVM,
keeping the comparison fair across models.

NOTE: unlike kNN and SVM, Random Forest is comparatively robust to
high-leverage points (Lecture 9) — the bagging procedure means no single
extreme point dominates all trees, only the subset of bootstrap samples
it happens to appear in. This is a genuine advantage over SVM, where
extreme points near the margin can become support vectors and directly
shift the decision boundary.

A deeper outlier/high-leverage analysis comparing which specific test
points all three models consistently misclassify is deferred to
06_evaluation.py, where all three models' predictions are available
simultaneously for comparison."""