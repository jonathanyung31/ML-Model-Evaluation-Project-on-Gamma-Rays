# =============================================================================
# 05_random_forest.py — Advanced Classifier 2: Random Forest
# =============================================================================

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
y_train = pd.read_csv(os.path.join("data", "processed", "y_train.csv")).squeeze()   # into pandas series
y_test = pd.read_csv(os.path.join("data", "processed", "y_test.csv")).squeeze()     # into pandas series

feature_names = X_train.columns.tolist()
# store feature names now, before any numpy conversion so labels don't get lost 
# before Section 7 where we need it to plot feature importance

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

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

n_estimators_values = [50, 100, 200, 300]   # B number of trees
max_depth_values = [None, 5, 10, 20]    # maximum level each tree could go to

# initialize with lowest possible value so any real score will replace it
best_score = -np.inf
best_params = {}    # to store best parameters
results = []    # to store all that was tested already

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
            scoring="balanced_accuracy"     # average score of TPR + TNR
        )

        # as each calibration runs 5 times (for 5 folds) we need its averaged stats
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
              f"balanced_acc={mean_score:.3f} (std={std_score:.3f}) (meaned error={mean_error:.3f})")

        if mean_score > best_score or (
            mean_score == best_score and n_est < best_params["n_estimators"]
        ):
            best_score = mean_score
            best_params = {"n_estimators": n_est, "max_depth": depth}

print(f"\nBest parameters: {best_params}")
print(f"Best CV balanced accuracy: {best_score:.3f}")


# =============================================================================
# SECTION 4: VISUALIZE HYPERPARAMETER SEARCH RESULTS
# =============================================================================


results_df = pd.DataFrame(results)  # turned into tables with labels

pivot = results_df.pivot(   # shaping it into matrix format
    index="max_depth",
    columns="n_estimators",
    values="mean_balanced_acc"
)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")

ax.set_xticks(range(len(pivot.columns)))    # adding number of ticks for x-axis
ax.set_xticklabels(pivot.columns)           # where should ticks appear at

ax.set_yticks(range(len(pivot.index)))      # adding number of ticks for y-axis
ax.set_yticklabels([str(d) for d in pivot.index])   # where ticks should appear at

ax.set_xlabel("n_estimators (number of trees)", fontsize=11)
ax.set_ylabel("max_depth", fontsize=11)
ax.set_title("Random Forest: Balanced Accuracy (CV)", fontsize=13)
fig.colorbar(im, ax=ax, label="Balanced Accuracy")

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        value = pivot.values[i, j]

        if value > 0.80:
            color = "black"
        else:
            color = "white"

        ax.text(j, i, f"{value:.3f}",
                ha="center", va="center",
                color=color, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "11_rf_hyperparameter_search.png"), dpi=150)
plt.show()
print("Plot saved: 11_rf_hyperparameter_search.png")


# =============================================================================
# SECTION 5: TRAIN FINAL MODEL ON FULL TRAINING SET, EVALUATE ON TEST SET
# =============================================================================

final_rf = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    max_features="sqrt",
    random_state=42,
    n_jobs=-1   # use all CPUs
)
final_rf.fit(X_train, y_train)

y_pred = final_rf.predict(X_test)
y_pred_proba = final_rf.predict_proba(X_test)[:, list(final_rf.classes_).index("g")]


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

# feature_importances_ is a built in attribute after training a Random Forest.
importances = final_rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]   # sort descending - most important first

print("\nFeature Importance (sorted)")
for i in sorted_idx:
    print(f"{feature_names[i]}: {importances[i]:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    range(len(importances)),
    importances[sorted_idx],
    color="darkorange",
    edgecolor="black"
)

ax.set_xticks(range(len(importances)))  # where the tick marks will be at
ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45, ha="right")

ax.set_ylabel("Feature Importance (mean Gini reduction)", fontsize=11)  # where the tick marks will be at
ax.set_title("Random Forest: Feature Importance", fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "13_rf_feature_importance.png"), dpi=150)
plt.show()
print("Plot saved: 13_rf_feature_importance.png")


# =============================================================================
# SECTION 8: SENSITIVITY, SPECIFICITY, auROC
# =============================================================================
"""Same label convention as 03_baseline_knn.py and 04_svm.py:
gamma (g) is the signal/positive class."""

tp = cm[0, 0]   # true g predicted as g
fn = cm[0, 1]   # true g predicted as h
fp = cm[1, 0]   # true h predicted as g
tn = cm[1, 1]   # true h predicted as h

sensitivity = tp / (tp + fn)    # recall or TPR
specificity = tn / (tn + fp)    # TPR

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

# g turns to 1, h turns to 0
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