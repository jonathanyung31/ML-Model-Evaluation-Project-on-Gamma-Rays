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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
# Wrapper-method feature selection (Lecture 6 Quick Summary table:
# "Backward elimination -- Start full, keep removing the worst feature";
# same method already used for kNN and SVM, applied with RF as the judge)
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    ConfusionMatrixDisplay
)
from tempfile import mkdtemp  # ADDED
from joblib import Memory  # ADDED

# ADDED: the feature_select step below uses a FIXED probe classifier
# (n_estimators=200, max_features="sqrt") that never changes across the
# grid search over clf__n_estimators/max_depth/max_features. Without
# caching, sklearn refits it from scratch for every one of the 48
# (n_estimators x max_depth x max_features) combinations, even though the
# result is identical every time. Memory caching computes it ONCE per CV
# fold and reuses that result for every combination evaluated on that
# fold -- this does not change the result in any way, it only removes
# redundant recomputation.
pipe_cache = Memory(location=mkdtemp(), verbose=0)  # ADDED

X_train = pd.read_csv(os.path.join("data", "processed", "X_train.csv"))
X_test = pd.read_csv(os.path.join("data", "processed", "X_test.csv"))
y_train = pd.read_csv(os.path.join("data", "processed", "y_train.csv")).squeeze()   # into pandas series
y_test = pd.read_csv(os.path.join("data", "processed", "y_test.csv")).squeeze()     # into pandas series

# REMOVED: feature_names = X_train.columns.tolist()   <- now set later, after feature selection picks the final subset (see Section 3+4b below)


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
   NOW TUNED (not fixed) below, since Lecture 9 lists it as a genuine
   parameter of the model, not a constant.

3. max_depth: how deep each tree is allowed to grow. Deeper trees =
   more flexible, higher overfitting risk (Lecture 3b, Section 2).
   None means trees grow until all leaves are pure or have fewer than
   min_samples_split samples.

4. The split criterion: Gini index or entropy (Lecture 9, Section 32).
   Both measure node impurity — lower impurity = better split. Gini is
   the default and generally performs similarly to entropy in practice."""  # CHANGED: point 2's note about tuning added


"""Lecture 10 ("Wrap Up"), Step 5 -- Shortlist Promising Models:
"If your dataset is huge, you might sample a smaller portion for this
step so training is fast. Just know that complex models (like Random
Forests...) suffer more from having less data." Same pattern already
used in 04_svm.py's X_train_sub. We use 6000 rows here instead of SVM's
4000, precisely because of that RF-specific caveat -- RF needs a bit more
data than SVM/kNN to behave representatively.

This subsample is used ONLY for the nested search below (Section 3+4b),
which is what was taking forever: it retrains the feature-selection probe
and the classifier repeatedly across every fold x every grid combination.
The FINAL model is explicitly refit on the FULL training set afterward
(Section 5), per Lecture 10 Step 6: "Use as much data as you can" for the
fine-tuned final model -- so nothing about the deployed model's quality
changes, only the search that finds its hyperparameters gets faster."""

X_train_sub, _, y_train_sub, _ = train_test_split(
    X_train, y_train,
    train_size=6000,
    stratify=y_train,
    random_state=42
)

print(f"\nSubsample for nested search: {X_train_sub.shape[0]} rows")

# =============================================================================
# SECTION 3+4b: HYPERPARAMETER TUNING + FEATURE SELECTION — NESTED TOGETHER  # CHANGED: sections merged/renamed
# =============================================================================

"""WHY NESTED, AND WHY TUNE max_features TOO?
Same nested-CV fix as the kNN/SVM scripts (Lecture 7, Mistake 2): feature
selection now lives inside the Pipeline that GridSearchCV cross-validates,
so it's refit per fold rather than done once beforehand using
hyperparameters chosen on the full (unselected) feature set.

We also now tune max_features (m) instead of fixing it at 'sqrt' for
every run — Lecture 9, Section 38 lists m as one of Random Forest's four
core parameters (alongside B, max_depth, and split criterion), with
sqrt(p) given only as the typical default, not a fixed constant."""  # CHANGED: docstring rewritten

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_rf = Pipeline([
    ("feature_select", SequentialFeatureSelector(
        RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=42, n_jobs=1),  # fixed probe  # ADDED
        n_features_to_select="auto",
        tol=0.001,
        direction="backward",
        scoring="balanced_accuracy",
        cv=cv_strategy,
        n_jobs=1 # CHANGED: was -1 -- outer GridSearchCV already parallelizes
                  # across grid points/folds; parallelizing here AND inside the
                  # probe RF too oversubscribes the CPU (was 4 nested levels of
                  # parallelism: GridSearchCV, SequentialFeatureSelector, the
                  # probe RF, and the final clf RF, all fighting for the same cores)
    )),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=1))
], memory=pipe_cache)  # cache feature_select's fit per fold (see note above)

param_grid = {  # was two separate lists `n_estimators_values` / `max_depth_values` used in a manual double for-loop
    "clf__n_estimators": [50, 100, 200, 300],  # was `n_estimators_values = [50, 100, 200, 300]`
    "clf__max_depth": [None, 5, 10, 20],  # was `max_depth_values = [None, 5, 10, 20]`
    "clf__max_features": ["sqrt", "log2", None],  # max_features is now tuned instead of fixed
}

grid_search = GridSearchCV(  # replaces the old manual double for-loop + cross_val_score entirely
    pipe_rf,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring="balanced_accuracy",
    n_jobs=-1
)
grid_search.fit(X_train_sub, y_train_sub)  # CHANGED: was grid_search.fit(X_train, y_train)

best_params = {  # was populated inside the manual for-loop via `if mean_score > best_score:` comparisons
    "n_estimators": grid_search.best_params_["clf__n_estimators"],
    "max_depth": grid_search.best_params_["clf__max_depth"],
    "max_features": grid_search.best_params_["clf__max_features"],
}

print(f"\nBest parameters: {best_params}")
print(f"Best CV balanced accuracy: {grid_search.best_score_:.3f}")  # CHANGED: was `best_score` (a plain float tracked manually)

best_pipeline = grid_search.best_estimator_
selected_mask = best_pipeline.named_steps["feature_select"].get_support()
selected_features = X_train.columns[selected_mask].tolist()  # was `X_train.columns[sfs.get_support()]`, run in a separate later step
dropped_features = X_train.columns[~selected_mask].tolist()  # was `X_train.columns[~sfs.get_support()]`
feature_names = selected_features  # was set from the un-selected `X_train.columns.tolist()` back in Section 1

print(f"\n--- Random Forest feature selection (wrapper: backward elimination) ---")
print(f"Selected features ({len(selected_features)}): {selected_features}")
print(f"Dropped features ({len(dropped_features)}): {dropped_features}")

# =============================================================================
# SECTION 4: VISUALIZE HYPERPARAMETER SEARCH (slice at the winning max_features)  # CHANGED: header + logic
# =============================================================================


results_df = pd.DataFrame(grid_search.cv_results_)  # CHANGED: was built manually from the `results` list appended inside the for-loop
slice_df = results_df[results_df["param_clf__max_features"] == best_params["max_features"]]  # ADDED — needed since the grid is now 3D (n_estimators x max_depth x max_features)

pivot = slice_df.pivot(  # was `results_df.pivot(...)`, now pivots the sliced 2D subset
    index="param_clf__max_depth", 
    columns="param_clf__n_estimators", 
    values="mean_test_score"  # was `index="max_depth", columns="n_estimators", values="mean_balanced_acc"`
)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")

ax.set_xticks(range(len(pivot.columns)))    # adding number of ticks for x-axis
ax.set_xticklabels(pivot.columns)           # where should ticks appear at

ax.set_yticks(range(len(pivot.index)))      # adding number of ticks for y-axis
ax.set_yticklabels([str(d) for d in pivot.index])   # where ticks should appear at

ax.set_xlabel("n_estimators (number of trees)", fontsize=11)
ax.set_ylabel("max_depth", fontsize=11)
ax.set_title(f"Random Forest: Balanced Accuracy (CV), max_features={best_params['max_features']}", fontsize=12)  # CHANGED: title now states which max_features slice is shown
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

# ADDED: refit explicitly on the FULL training set (not just the 6000-row
# search subsample), per Lecture 10 Step 6: "Use as much data as you can
# at this stage." Same pattern as 04_svm.py's final_svm refit.
final_rf = RandomForestClassifier(  # CHANGED: was `final_rf = best_pipeline.named_steps["clf"]`
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    max_features=best_params["max_features"],
    random_state=42,
    n_jobs=-1   # ADDED: fine to parallelize now -- this is a single fit, not nested inside a search
)

X_train = X_train[selected_features]   # keep for feature_importances_ / plotting below  # ADDED
X_test = X_test[selected_features]

final_rf.fit(X_train, y_train)  # ADDED: explicit fit on the FULL 15216-row training set

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
"""Unlike an earlier draft's comment here, this Random Forest DOES use
wrapper-based backward elimination (Section 3+4b above), now properly
nested inside the same CV folds used for hyperparameter tuning — it is
NOT run on all 10 features unselected.

Random Forest is comparatively robust to high-leverage points (Lecture
9) — bagging means no single extreme point dominates every tree, only
the bootstrap samples it happens to land in. This is a genuine advantage
over SVM, where extreme points near the margin can become support
vectors and directly shift the decision boundary.

A deeper outlier/high-leverage analysis comparing which specific test
points all three models consistently misclassify is deferred to
06_evaluation.py, where all three models' predictions are available
simultaneously for comparison."""  # CHANGED: opening paragraph rewritten to stop contradicting Section 3+4b

# =============================================================================
# SECTION 10b: MISCLASSIFIED POINTS RANKED BY CONFIDENCE (Lecture 7)
# =============================================================================
"""RF has no geometric margin, so predict_proba() (fraction of trees
voting for the predicted class) is the closest analogue to Lecture 7's
"distance to boundary" guidance."""

rf_confidence = final_rf.predict_proba(X_test).max(axis=1)
misclassified_mask = (y_pred != y_test.values)

rf_confidence_report = pd.DataFrame({
    "true_class": y_test.values[misclassified_mask],
    "predicted_class": y_pred[misclassified_mask],
    "model_confidence": rf_confidence[misclassified_mask],
}, index=X_test.index[misclassified_mask]).sort_values(
    "model_confidence", ascending=False)

print(f"\n--- {misclassified_mask.sum()} misclassified test points out of {len(y_test)} ---")
print("Top 5 most confidently WRONG predictions:")
print(rf_confidence_report.head(5))

# =============================================================================
# SECTION 11: SAVE RESULTS FOR 06_evaluation.py  # ADDED
# =============================================================================

os.makedirs(os.path.join("results", "model_outputs"), exist_ok=True)  # ADDED

np.savez(  # ADDED
    os.path.join("results", "model_outputs", "rf_results.npz"),
    fpr=fpr,
    tpr=tpr,
    auroc=auroc,
    sensitivity=sensitivity,
    specificity=specificity,
    selected_features=np.array(selected_features, dtype=object),
    feature_importances=importances,        # ADDED — RF-specific, for the Step 6 discussion
    feature_names=np.array(feature_names, dtype=object),
    misclassified_indices=rf_confidence_report.index.to_numpy(),  # add this line
)  # ADDED

print("Results saved: results/model_outputs/rf_results.npz")  # ADDED