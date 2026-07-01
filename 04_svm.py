# =============================================================================
# 04_svm.py — Advanced Classifier 1: Support Vector Machine
# =============================================================================
"""WHAT IS THIS FILE?
This is the fourth step of the project: training our first advanced
classifier, an SVM with an RBF kernel. SVM is "advanced" relative to our
kNN baseline because it requires hyperparameter optimization over more
than one setting (C and gamma) and uses the kernel trick (Lecture 9) to
draw non-linear decision boundaries.

WHY SVM FOR THIS DATASET?
The MAGIC features (Hillas parameters) are continuous, and Lecture 9
describes SVM with RBF kernel as well-suited to this kind of data. SVM
also naturally produces a continuous decision score (Lecture 9, Section
20), which lets us build ROC curves directly — exactly what the dataset
documentation requires for comparing classifiers.

WHY DOES SVM NEED NORMALIZED DATA?
Like kNN, SVM's hyperplane and especially its RBF kernel rely on distance
between points (Lecture 9, Section 18: the RBF kernel is literally a
function of squared distance). Unscaled features would distort the
margin and kernel calculations the same way they would distort kNN's
distance calculations. This is why we use the already-normalized data
from 02_preprocessing.py."""

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
"""SVM training time grows quickly with the number of samples, especially
with an RBF kernel. With ~15000 training rows and a grid search over many
(C, gamma) combinations times 5 CV folds, training on the full set would
be very slow. Lecture 10 (Step 5) explicitly allows this: "If your dataset
is huge, you might sample a smaller portion for this step so training is
fast." We use a stratified subsample ONLY for the hyperparameter search
below — the FINAL model in Section 5 is retrained on the full training
set with the best hyperparameters found here."""

from sklearn.model_selection import train_test_split

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
"""SVM with RBF kernel has two hyperparameters (Lecture 9, Sections 12-18):
C: controls the tradeoff between a wide margin and classifying every
   training point correctly. Large C = narrow margin, tries hard to
   classify everything correctly (risk of overfitting). Small C = wider
   margin, more tolerant of misclassified points (more generalizable).
gamma: controls how local/tight the RBF kernel's influence is. Large
   gamma = each point only strongly influences its immediate neighborhood
   (risk of overfitting). Small gamma = smoother, more global influence.

WHY THESE SPECIFIC VALUES?
Lecture 9 explicitly recommends testing values in exponential steps
(0.01, 0.1, 1, 10, 100) rather than a fine linear grid, since both C and
gamma act on a multiplicative, not additive, scale. We use a coarse
exponential grid first (multilevel grid search, Lecture 8, Method 2),
then could zoom in around the best region if needed."""

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1]
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

"""WHY GridSearchCV?
This performs an exhaustive search (Lecture 8, Method 1) over every
combination in param_grid — 4 x 4 = 16 combinations — using cross-
validation on the (sub-sampled) training data only, exactly as Lecture
3b/7 requires: no test data is touched during this tuning step."""

grid_search = GridSearchCV(
    SVC(kernel="rbf", probability=True, random_state=42),
    param_grid=param_grid,
    cv=cv_strategy,
    scoring="balanced_accuracy",
    n_jobs=-1
)

grid_search.fit(X_train_sub, y_train_sub)

print("\n--- Grid search results ---")
print(f"Best C: {grid_search.best_params_['C']}")
print(f"Best gamma: {grid_search.best_params_['gamma']}")
print(f"Best CV balanced accuracy: {grid_search.best_score_:.3f}")

best_C = grid_search.best_params_["C"]
best_gamma = grid_search.best_params_["gamma"]


# =============================================================================
# SECTION 4: VISUALIZE THE GRID SEARCH RESULTS
# =============================================================================
"""This heatmap shows balanced accuracy across every (C, gamma) combination
tested, making it visually clear which region of the hyperparameter space
performs best — and whether the best point sits in the middle of the grid
(good) or at an edge (a sign we may need to extend the search range)."""

results_df = pd.DataFrame(grid_search.cv_results_)
pivot_table = results_df.pivot(
    index="param_gamma", columns="param_C", values="mean_test_score"
)

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(pivot_table.values, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(pivot_table.columns)))
ax.set_xticklabels(pivot_table.columns)
ax.set_yticks(range(len(pivot_table.index)))
ax.set_yticklabels(pivot_table.index)
ax.set_xlabel("C", fontsize=11)
ax.set_ylabel("gamma", fontsize=11)
ax.set_title("SVM Grid Search: Balanced Accuracy", fontsize=13)
fig.colorbar(im, ax=ax, label="Balanced Accuracy")

for i in range(len(pivot_table.index)):
    for j in range(len(pivot_table.columns)):
        ax.text(j, i, f"{pivot_table.values[i, j]:.3f}",
                ha="center", va="center", color="white", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "08_svm_grid_search.png"), dpi=150)
plt.show()
print("Plot saved: 08_svm_grid_search.png")

"""WHAT TO LOOK FOR:
The best (C, gamma) combination should not sit exactly on the edge of the
grid — if it does, the true optimum may lie outside our tested range, and
we would need to extend param_grid further in that direction."""


# =============================================================================
# SECTION 5: TRAIN FINAL MODEL ON FULL TRAINING SET, EVALUATE ON TEST SET
# =============================================================================
"""Now we retrain using the best (C, gamma) found above, but on the FULL
training set (not the subsample used only for the search) — using all
available training data for the final model, as Lecture 10 (Step 6)
recommends: "Use as much data as you can at this stage." Only after this
do we touch the test set, for the one and only honest evaluation."""

final_svm = SVC(kernel="rbf", C=best_C, gamma=best_gamma, probability=True, random_state=42)
final_svm.fit(X_train, y_train)

y_pred = final_svm.predict(X_test)
y_pred_proba = final_svm.predict_proba(X_test)[:, list(final_svm.classes_).index("g")]

"""WHY probability=True AND predict_proba?
Same reasoning as 03_baseline_knn.py: we need a continuous score per
point (not just a hard label) to build the ROC curve and compute auROC
(Lecture 9, Section 20)."""


# =============================================================================
# SECTION 6: CONFUSION MATRIX (ABSOLUTE AND RELATIVE)
# =============================================================================
"""Same structure as 03_baseline_knn.py, so the two models can be directly
compared in 06_evaluation.py."""

cm = confusion_matrix(y_test, y_pred, labels=["g", "h"])
cm_relative = cm.astype(float) / cm.sum(axis=1, keepdims=True)

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
# SECTION 8: ROC CURVE AND REPORTING AT REQUIRED FPR THRESHOLDS
# =============================================================================
"""Same structure as 03_baseline_knn.py, using the same FPR thresholds
specified in the dataset documentation (magic04.names), so kNN and SVM
results line up directly in 06_evaluation.py."""

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