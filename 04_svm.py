# =============================================================================
# 04_svm.py — Advanced Classifier 1: Support Vector Machine
# =============================================================================

# =============================================================================
# SECTION 1: IMPORTS AND LOAD PREPROCESSED DATA
# =============================================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    ConfusionMatrixDisplay
)

X_train = pd.read_csv(os.path.join("data", "processed", "X_train.csv"))
X_test = pd.read_csv(os.path.join("data", "processed", "X_test.csv"))
y_train = pd.read_csv(os.path.join("data", "processed", "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join("data", "processed", "y_test.csv")).squeeze()

print("Loaded preprocessed data:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")


# =============================================================================
# SECTION 2: SUBSAMPLE FOR HYPERPARAMETER SEARCH (PRACTICAL NECESSITY)
# =============================================================================

from sklearn.model_selection import train_test_split

# We subsample only for the training dataset
X_train_sub, _, y_train_sub, _ = train_test_split(
    X_train, y_train,
    train_size=4000,
    stratify=y_train,
    random_state=42
)

print(f"\nSubsample for hyperparameter search: {X_train_sub.shape[0]} rows")


# =============================================================================
# SECTION 3: HYPERPARAMETER SEARCH — C AND GAMMA (MULTILEVEL GRID SEARCH)
# =============================================================================

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1]
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# specifically made for hyperparameter tunning using grid search
grid_search = GridSearchCV(     
    SVC(kernel="rbf", probability=True, random_state=42),   # allows prob. - helps auROC
    param_grid=param_grid,
    cv=cv_strategy,
    scoring="balanced_accuracy",    # average score of TPR + TNR
    n_jobs=-1   # using all cpu cores to make computations much faster
)

grid_search.fit(X_train_sub, y_train_sub)   # run the grid

print("\n--- Grid search results ---")
print(f"Best C: {grid_search.best_params_['C']}")
print(f"Best gamma: {grid_search.best_params_['gamma']}")

# best average CV score based on balanced accuracy
print(f"Best CV balanced accuracy: {grid_search.best_score_:.3f}")

best_C = grid_search.best_params_["C"]
best_gamma = grid_search.best_params_["gamma"]


# =============================================================================
# SECTION 4: VISUALIZE THE GRID SEARCH RESULTS
# =============================================================================

results_df = pd.DataFrame(grid_search.cv_results_)
pivot_table = results_df.pivot(
    index="param_gamma", columns="param_C", values="mean_test_score"
)

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(pivot_table.values, cmap="viridis", aspect="auto")    # displays as heatmap

ax.set_xticks(range(len(pivot_table.columns)))  # tick mark with numbers at every column
ax.set_xticklabels(pivot_table.columns)     # replacing tese numbers with C values

ax.set_yticks(range(len(pivot_table.index)))
ax.set_yticklabels(pivot_table.index)

ax.set_xlabel("C", fontsize=11)
ax.set_ylabel("gamma", fontsize=11)
ax.set_title("SVM Grid Search: Balanced Accuracy", fontsize=13)
fig.colorbar(im, ax=ax, label="Balanced Accuracy")

for i in range(len(pivot_table.index)):
    for j in range(len(pivot_table.columns)):
        value = pivot_table.values[i, j]
        if value > 0.75:
            color = "black"
        else:
            color = "white"
        ax.text(j, i, f"{value:.3f}",
                ha="center", va="center",
                color=color, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "08_svm_grid_search.png"), dpi=150)
plt.show()
print("Plot saved: 08_svm_grid_search.png")


# =============================================================================
# SECTION 5: TRAIN FINAL MODEL ON FULL TRAINING SET, EVALUATE ON TEST SET
# =============================================================================

final_svm = SVC(kernel="rbf", C=best_C, gamma=best_gamma, probability=True, random_state=42)
final_svm.fit(X_train, y_train)

y_pred = final_svm.predict(X_test)
y_pred_proba = final_svm.predict_proba(X_test)[:, list(final_svm.classes_).index("g")]

# =============================================================================
# SECTION 6: CONFUSION MATRIX (ABSOLUTE AND RELATIVE)
# =============================================================================
"""Same structure as 03_baseline_knn.py, so the two models can be directly
compared in 06_evaluation.py."""

cm = confusion_matrix(y_test, y_pred, labels=["g", "h"])
cm_relative = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # divide by total counts of each row

print("\n--- Confusion Matrix (absolute) ---")
print(cm)

print("\n--- Confusion Matrix (relative, per true class) ---")
print(np.round(cm_relative, 3))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ConfusionMatrixDisplay(cm, display_labels=["g", "h"]).plot(ax=axes[0], cmap="Greens")
axes[0].set_title("Absolute Confusion Matrix")

ConfusionMatrixDisplay(cm_relative, display_labels=["g", "h"]).plot(ax=axes[1], cmap="Greens")
axes[1].set_title("Relative Confusion Matrix")

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "09_svm_confusion_matrix.png"), dpi=150)
plt.show()
print("Plot saved: 09_svm_confusion_matrix.png")


# =============================================================================
# SECTION 7: SENSITIVITY, SPECIFICITY, auROC
# =============================================================================
"""Same label convention as 03_baseline_knn.py: gamma (g) is the
signal/positive class, matching the dataset's own framing."""

tp = cm[0, 0]   # true g predicted as g
fn = cm[0, 1]   # true g predicted as h
fp = cm[1, 0]   # true h predicted as g
tn = cm[1, 1]   # true h predicted as h

sensitivity = tp / (tp + fn)    # recall (TPR)
specificity = tn / (tn + fp)    # TNR

print(f"\nSensitivity (gamma correctly identified): {sensitivity:.3f}")
print(f"Specificity (hadron correctly identified): {specificity:.3f}")

auroc = roc_auc_score((y_test == "g").astype(int), y_pred_proba)
print(f"auROC: {auroc:.3f}")


# =============================================================================
# SECTION 8: ROC CURVE AND REPORTING AT REQUIRED FPR THRESHOLDS
# =============================================================================

fpr, tpr, thresholds = roc_curve((y_test == "g").astype(int), y_pred_proba)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color="seagreen", label=f"SVM RBF (auROC = {auroc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guessing")
ax.set_xlabel("False Positive Rate (1 - specificity)", fontsize=11)
ax.set_ylabel("True Positive Rate (sensitivity)", fontsize=11)
ax.set_title("SVM (RBF Kernel) ROC Curve", fontsize=13)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "10_svm_roc_curve.png"), dpi=150)
plt.show()
print("Plot saved: 10_svm_roc_curve.png")

required_fpr_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]

print("\n--- Sensitivity at required FPR thresholds (per dataset documentation) ---")
# showing the changing relationship between TPR and FPR through the ROC with specific cuttoffs
for target_fpr in required_fpr_thresholds:
    idx = np.argmin(np.abs(fpr - target_fpr))
    print(f"FPR ~{target_fpr}: sensitivity = {tpr[idx]:.3f} (actual FPR at this point: {fpr[idx]:.3f})")


# =============================================================================
# SECTION 9: WHY NOT TRY A LINEAR KERNEL FIRST? (REASONING NOTE)
# =============================================================================
"""Lecture 9 suggests starting with a linear SVM to check if linear
separability exists before moving to RBF. Our EDA (01_eda.py) already
gives indirect evidence against pure linear separability: even our
strongest individual feature (fAlpha) only separates classes partially,
and the 2D scatter plots showed curved, overlapping boundaries rather
than anything resembling a straight line. We therefore went directly to
the RBF kernel, but this assumption could be explicitly tested by also
fitting SVC(kernel="linear") and comparing its auROC against the RBF
result here, if time allows."""


# =============================================================================
# SECTION 10: NOTE ON FEATURE SELECTION AND HIGH LEVERAGE / OUTLIERS
# =============================================================================
"""Per Lecture 3b (mistake #2) and Lecture 7, feature selection and
outlier/high-leverage analysis must be model-specific. For this SVM
model, we use all 10 features without selection, consistent with our
kNN baseline, so the two models are compared on equal footing. SVM is
also known to be sensitive to high-leverage points (Lecture 7: such
points can pull the decision boundary), since its boundary depends only
on the support vectors — points near the margin. A deeper pass
identifying which test points are misclassified with the largest margin
violations is left for 06_evaluation.py if time allows."""