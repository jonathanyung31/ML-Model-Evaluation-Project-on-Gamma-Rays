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


# SECTION 2: CHOOSE k VALUES TO TEST (HYPERPARAMETER TUNING)


k_values = list(range(1, 21))

# prevent noise with stratification
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# stores cross-validated error values for each k to find best hyperparameter
cv_errors = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        knn, X_train, y_train,
        cv=cv_strategy,
        scoring="balanced_accuracy"     # average score of TPR + TNR
    )
    mean_error = 1 - scores.mean()   # convert accuracy to an error rate
    cv_errors.append(mean_error)
    print(f"k={k}: mean CV balanced-accuracy error = {mean_error:.3f}")


# SECTION 3: PLOT VALIDATION ERROR vs k


fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(k_values, cv_errors, marker="o", color="steelblue")     # dots on each data point
ax.set_xlabel("k (number of neighbors)", fontsize=11)
ax.set_ylabel("Cross-validated error (1 - balanced accuracy)", fontsize=11)
ax.set_title("kNN: Validation Error vs. k", fontsize=13)
ax.grid(True)   # turn on grid lines for easier comparisons

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "05_knn_k_selection.png"), dpi=150)
plt.show()
print("Plot saved: 05_knn_k_selection.png")

best_k = k_values[np.argmin(cv_errors)]     # finding lowest error value
print(f"\nBest k selected via cross-validation: {best_k}")


# SECTION 4: TRAIN FINAL MODEL WITH BEST k, EVALUATE ON TEST SET

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)

y_pred = final_knn.predict(X_test)

# will return probabilities for each class
y_pred_proba = final_knn.predict_proba(X_test)[:, list(final_knn.classes_).index("g")]


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
# showing the changing relationship between TPR and FPR through the ROC with specific cuttoffs
for target_fpr in required_fpr_thresholds:
    idx = np.argmin(np.abs(fpr - target_fpr))
    print(f"FPR ~{target_fpr} (closest available point: {fpr[idx]:.3f}): sensitivity = {tpr[idx]:.3f}")