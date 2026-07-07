# =============================================================================
# 03_baseline_knn.py — Base Experiment: k-Nearest Neighbors
# =============================================================================


# SECTION 1: IMPORTS AND LOAD PREPROCESSED DATA

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
# Import tools for cross-validation and stratified k-fold splitting
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV  
from sklearn.pipeline import Pipeline
from tempfile import mkdtemp  # ADDED
from joblib import Memory  # ADDED
# ADDED: the feature_select step below uses a FIXED probe classifier
# (n_neighbors=15) that never changes across the grid search over
# clf__n_neighbors. Without caching, sklearn refits it from scratch for
# every one of the 20 candidate k values, even though the result is
# identical every time. Memory caching computes it ONCE per CV fold and
# reuses that result for every k evaluated on that fold -- this does not
# change the result in any way, it only removes redundant recomputation.
pipe_cache = Memory(location=mkdtemp(), verbose=0)  # ADDED

# Wrapper-method feature selection (Lecture 6, Method 3: backward elimination)
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
# Import functions for evaluating classification models
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    ConfusionMatrixDisplay
)


plt.style.use("dark_background")
plt.rcParams["grid.color"] = "#3a3a3a"
plt.rcParams["grid.alpha"] = 0.6


X_train = pd.read_csv(os.path.join("data", "processed", "X_train.csv"))
X_test = pd.read_csv(os.path.join("data", "processed", "X_test.csv"))
y_train = pd.read_csv(os.path.join("data", "processed", "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join("data", "processed", "y_test.csv")).squeeze()

print("Loaded preprocessed data:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# prevent noise with stratification -- shared by feature selection AND
# the k-search below, so both stages see the exact same folds
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# =============================================================================
# SECTION 1b: FEATURE SELECTION -- WRAPPER METHOD (BACKWARD ELIMINATION)
# =============================================================================
"""WHY NESTED, AND WHY FOR kNN SPECIFICALLY?
Lecture 7 ('Mistake 2'): doing feature selection on the whole training
set and then reusing the same folds to tune a hyperparameter lets those
folds indirectly influence the feature choice before they're used as
validation folds. Fix (same lecture): 'redo feature selection
independently inside each fold.' Wrapping the selector and the
classifier in one Pipeline and handing that to GridSearchCV does exactly
this — for every fold, sklearn refits the feature selector on ONLY that
fold's training partition before scoring the candidate k on the held-out
partition.

kNN is also the model most directly hurt by irrelevant/noisy features:
every feature enters the Euclidean distance equally, so noise features
dilute the true signal and can trigger the curse of dimensionality
(Lecture 4 Section 23, Lecture 5b ISLR 4.8)."""

pipe_knn = Pipeline([
    ("scaler", StandardScaler()),  # ADDED: refit per fold now, fixes Mistake 1 at the inner-CV level
    ("feature_select", SequentialFeatureSelector(
        KNeighborsClassifier(n_neighbors=15),
        n_features_to_select="auto",
        tol=0.001,      # tolerance for stopping automatic feature selection (when to stop)
        direction="backward",
        scoring="balanced_accuracy",
        cv=cv_strategy,
        n_jobs=1  # 1 cpu works on this
    )),
    ("clf", KNeighborsClassifier())
], memory=pipe_cache)  # ADDED: cache feature_select's fit per fold (see note above)

k_values = list(range(1, 21))

grid_search = GridSearchCV(
    pipe_knn,
    param_grid={"clf__n_neighbors": k_values},
    cv=cv_strategy,
    scoring="balanced_accuracy",
    n_jobs=-1   # use all cpus
)
grid_search.fit(X_train, y_train)

results_df = pd.DataFrame(grid_search.cv_results_)
cv_errors = (1 - results_df["mean_test_score"]).tolist()

best_k = grid_search.best_params_["clf__n_neighbors"]  # CHANGED: was `k_values[np.argmin(cv_errors)]` computed later
print(f"\nBest k selected via nested cross-validation: {best_k}")  # CHANGED: text updated to say "nested"

# GridSearchCV(refit=True) already retrained the winning pipeline on the
# FULL training set — this is normal, non-leaky final-model fitting,
# distinct from the per-fold tuning above.
best_pipeline = grid_search.best_estimator_  # ADDED
selected_mask = best_pipeline.named_steps["feature_select"].get_support()  # ADDED
selected_features = X_train.columns[selected_mask].tolist()  # CHANGED: was `X_train.columns[sfs.get_support()]`
dropped_features = X_train.columns[~selected_mask].tolist()  # CHANGED: was `X_train.columns[~sfs.get_support()]`

print(f"\n--- kNN feature selection (wrapper: backward elimination) ---")
print(f"Selected features ({len(selected_features)}): {selected_features}")
print(f"Dropped features ({len(dropped_features)}): {dropped_features}")

# REMOVED: X_train = X_train[selected_features]   <- no longer needed, Pipeline handles this internally
# REMOVED: X_test = X_test[selected_features]      <- no longer needed, Pipeline handles this internally

# SECTION 3: PLOT VALIDATION ERROR vs k


fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(k_values, cv_errors, marker="o", color="steelblue")     # dots on each data point
ax.set_xlabel("k (number of neighbors)", fontsize=11)
ax.set_ylabel("Cross-validated error (1 - balanced accuracy)", fontsize=11)
ax.set_title("kNN: Validation Error vs. k", fontsize=13)
ax.set_xticks(k_values)          # ADD THIS — force a tick at every k value, 1 through 20
ax.grid(True)   # turn on grid lines for easier comparisons

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "05_knn_k_selection.png"), dpi=150)
plt.show()
print("Plot saved: 05_knn_k_selection.png")

# REMOVED: best_k = k_values[np.argmin(cv_errors)]   <- moved up into Section 1b+2, computed from grid_search instead


# SECTION 4: TRAIN FINAL MODEL WITH BEST k, EVALUATE ON TEST SET

final_knn = best_pipeline  # CHANGED: was `KNeighborsClassifier(n_neighbors=best_k)` + separate `.fit(X_train, y_train)`

y_pred = final_knn.predict(X_test)  # CHANGED: X_test is now the FULL feature set, Pipeline selects internally

# will return probabilities for each class
y_pred_proba = final_knn.predict_proba(X_test)[:, list(final_knn.named_steps["clf"].classes_).index("g")]  # CHANGED: `.classes_` now via `.named_steps["clf"]` since final_knn is a Pipeline


# SECTION 5: CONFUSION MATRIX (ABSOLUTE AND RELATIVE)

cm = confusion_matrix(y_test, y_pred, labels=["g", "h"])
cm_relative = cm.astype(float) / cm.sum(axis=1, keepdims=True)

print("\nConfusion Matrix (absolute)")
print(cm)

print("\nConfusion Matrix (relative, per true class)")
print(np.round(cm_relative, 3))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ConfusionMatrixDisplay(cm, display_labels=["g", "h"]).plot(ax=axes[0], cmap="Blues")
axes[0].set_title("Absolute Confusion Matrix Heatmap")

ConfusionMatrixDisplay(cm_relative, display_labels=["g", "h"]).plot(ax=axes[1], cmap="Blues")
axes[1].set_title("Relative Confusion Matrix")

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "06_knn_confusion_matrix.png"), dpi=150)
plt.show()
print("Plot saved: 06_knn_confusion_matrix.png")


# SECTION 6: SENSITIVITY, SPECIFICITY, auROC

tp = cm[0, 0]   # g predicted as g
fn = cm[0, 1]   # g predicted as h
fp = cm[1, 0]   # h predicted as g
tn = cm[1, 1]   # h predicted as hS

sensitivity = tp / (tp + fn)    # recall (TPR)
specificity = tn / (tn + fp)    # TNR

print(f"\nSensitivity (gamma correctly identified): {sensitivity:.3f}")
print(f"Specificity (hadron correctly identified): {specificity:.3f}")

auroc = roc_auc_score((y_test == "g").astype(int), y_pred_proba)
print(f"auROC: {auroc:.3f}")


# SECTION 7: ROC CURVE AND REPORTING AT REQUIRED FPR THRESHOLDS


fpr, tpr, thresholds = roc_curve((y_test == "g").astype(int), y_pred_proba)

fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(fpr, tpr, color="deepskyblue", label=f"kNN (auROC = {auroc:.3f})")

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
# showing the changing relationship between TPR and FPR through the ROC with specific cuttoffs
for target_fpr in required_fpr_thresholds:
    idx = np.argmin(np.abs(fpr - target_fpr))
    print(f"FPR ~{target_fpr} (closest available point: {fpr[idx]:.3f}): sensitivity = {tpr[idx]:.3f}")

# =============================================================================
# SECTION 7b: MISCLASSIFIED POINTS RANKED BY CONFIDENCE (Lecture 7)
# =============================================================================
"""Lecture 7: "look at a sorted list of misclassified samples -- the
ones farthest from the boundary are your biggest problem cases."
kNN has no explicit boundary, so we use predict_proba() (the fraction
of the k neighbors that voted for the predicted class) as its
confidence measure. kNN is also the model most sensitive to
high-leverage points among the three, since distance is all it uses --
an unusual point can also corrupt its neighbors' predictions by being
their nearest neighbor."""

knn_confidence = final_knn.predict_proba(X_test).max(axis=1)
misclassified_mask = (y_pred != y_test.values)

knn_confidence_report = pd.DataFrame({
    "true_class": y_test.values[misclassified_mask],
    "predicted_class": y_pred[misclassified_mask],
    "model_confidence": knn_confidence[misclassified_mask],
}, index=X_test.index[misclassified_mask]).sort_values(
    "model_confidence", ascending=False)

print(f"\n--- {misclassified_mask.sum()} misclassified test points out of {len(y_test)} ---")
print("Top 5 most confidently WRONG predictions:")
print(knn_confidence_report.head(5))

# =============================================================================
# SECTION 8: SAVE RESULTS FOR 06_evaluation.py  # ADDED
# =============================================================================
"""06_evaluation.py runs as a separate process, so none of this script's
in-memory variables (fpr, tpr, auroc, ...) exist by the time it runs.
We save exactly what Project.md Step 6 needs: the ROC curve (for the
overlay plot) and auROC/sensitivity-at-thresholds (for the comparison
table)."""

os.makedirs(os.path.join("results", "model_outputs"), exist_ok=True)  # ADDED

np.savez(  # ADDED
    os.path.join("results", "model_outputs", "knn_results.npz"),
    fpr=fpr,
    tpr=tpr,
    auroc=auroc,
    sensitivity=sensitivity,
    specificity=specificity,
    selected_features=np.array(selected_features, dtype=object),
    misclassified_indices=knn_confidence_report.index.to_numpy(),  # add this line
)  # ADDED

print("Results saved: results/model_outputs/knn_results.npz")  # ADDED