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
from sklearn.pipeline import Pipeline  
# Wrapper-method feature selection (Lecture 6 Quick Summary table:
# "Backward elimination -- Start full, keep removing the worst feature")
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    ConfusionMatrixDisplay
)
from tempfile import mkdtemp
from joblib import Memory

# SPEED FIX: the feature_select step below uses a FIXED probe classifier
# (C=1, gamma="scale") that never changes across the grid search. Without
# caching, sklearn still refits it from scratch for every single (C, gamma)
# combination in the grid, even though the result is identical every time.
# Memory caching makes sklearn compute it ONCE per CV fold and reuse that
# result for every grid point evaluated on that same fold -- this does not
# change the result in any way, it only removes redundant recomputation.
pipe_cache = Memory(location=mkdtemp(), verbose=0)

X_train = pd.read_csv(os.path.join("data", "processed", "X_train.csv"))
X_test = pd.read_csv(os.path.join("data", "processed", "X_test.csv"))
y_train = pd.read_csv(os.path.join("data", "processed", "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join("data", "processed", "y_test.csv")).squeeze()

print("Loaded preprocessed data:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



plt.style.use("dark_background")
plt.rcParams["grid.color"] = "#3a3a3a"
plt.rcParams["grid.alpha"] = 0.6


from matplotlib.colors import ListedColormap
class_cmap = ListedColormap(["deepskyblue", "magenta"])

# =============================================================================
# SECTION 2: SUBSAMPLE FOR HYPERPARAMETER SEARCH (PRACTICAL NECESSITY)
# =============================================================================

from sklearn.model_selection import train_test_split

X_train_sub, _, y_train_sub, _ = train_test_split(
    X_train, y_train,
    train_size=4000,
    stratify=y_train,
    random_state=42
)

print(f"\nSubsample for hyperparameter search: {X_train_sub.shape[0]} rows")

# =============================================================================
# SECTION 2b: LINEAR SVM CHECK -- IS LINEAR SEPARABILITY ENOUGH?  # ADDED
# =============================================================================
"""Project.md Step 4: "Start with linear SVM to check if linear
separability exists" -- before committing to the RBF kernel's extra
complexity and hyperparameter search, check whether a plain linear
boundary already does reasonably well.

Tested on the SAME subsample and SAME CV folds used for the RBF search
below, on the ORIGINAL (unselected) features -- this is a diagnostic
that happens BEFORE feature selection, since the question here is
whether the raw feature space is linearly separable at all, not
whether a particular subset is. The result is compared against the
tuned RBF score once that search finishes (see Section 3 below), and
that comparison is the actual justification for using RBF -- not
just an assumption."""  # ADDED

from sklearn.model_selection import cross_val_score  # ADDED
from sklearn.svm import SVC as _SVC_for_linear_check  # ADDED (avoids shadowing the SVC import used below)

# CHANGED: wrap in a small Pipeline so scaling is refit per fold here too,
# not just in the main pipe_svm below -- this diagnostic uses cv_strategy
# just like the real search, so it needs the same fix.
linear_check_pipe = Pipeline([  # ADDED
    ("scaler", StandardScaler()),  # ADDED
    ("clf", _SVC_for_linear_check(kernel="linear", C=1, random_state=42))  # ADDED
])  # ADDED

linear_svm_scores = cross_val_score(
    linear_check_pipe,  # CHANGED: was `_SVC_for_linear_check(kernel="linear", C=1, random_state=42)`
    X_train_sub, y_train_sub,
    cv=cv_strategy,
    scoring="balanced_accuracy",
    n_jobs=-1
)
linear_svm_cv_score = linear_svm_scores.mean()

print(f"\n--- Linear SVM check (Project.md Step 4) ---")  # ADDED
print(f"Linear SVM CV balanced accuracy: {linear_svm_cv_score:.3f}")  # ADDED
print("(Compared against the tuned RBF score once the grid search below finishes.)")  # ADDED

# =============================================================================
# SECTION 2b+3: FEATURE SELECTION + C/GAMMA — NESTED, TRUE MULTILEVEL SEARCH  # CHANGED: section renamed/merged
# =============================================================================
"""WHY NESTED, AND WHY A REAL MULTILEVEL SEARCH?
Same nested-CV fix as 03_baseline_knn.py (Lecture 7, Mistake 2): the
feature selector now lives inside the Pipeline that GridSearchCV
cross-validates, so it's refit per fold instead of once on the whole
training set.

This also replaces the old single flat grid with an actual MULTILEVEL
grid search (Lecture 8, Method 2: coarse grid -> find best region ->
zoom in with a finer grid), rather than the single-pass exhaustive
search (Lecture 8, Method 1) the previous version actually implemented
despite being labeled 'multilevel'."""

pipe_svm = Pipeline([
    ("scaler", StandardScaler()),  # ADDED: refit per fold, fixes Mistake 1 for both coarse and fine grid search
    ("feature_select", SequentialFeatureSelector(
        SVC(kernel="rbf", C=1, gamma="scale", random_state=42),  # fixed probe
        n_features_to_select="auto",
        tol=0.001,
        direction="backward",
        scoring="balanced_accuracy",
        cv=cv_strategy,
        n_jobs=1    # SPEED FIX: outer GridSearchCV already parallelizes across
                  # grid points/folds; parallelizing here too oversubscribes
                  # the CPU (more processes than cores fighting each other)
    )),
    ("clf", SVC(kernel="rbf", probability=True, random_state=42))
], memory=pipe_cache)  # SPEED FIX: cache feature_select's fit per fold (see note above)


# ---- Level 1: coarse grid over a wide range ----
coarse_grid = {
    "clf__C": [0.01, 0.1, 1, 10, 100, 1000],
    "clf__gamma": [0.0001, 0.001, 0.01, 0.1, 1, 10]  
}  

coarse_search = GridSearchCV(  
    pipe_svm, param_grid=coarse_grid,  
    cv=cv_strategy, scoring="balanced_accuracy", n_jobs=-1  
)  
coarse_search.fit(X_train_sub, y_train_sub)  

coarse_C = coarse_search.best_params_["clf__C"]  # ADDED
coarse_gamma = coarse_search.best_params_["clf__gamma"]  # ADDED
print("\n--- Coarse grid search results ---")  # ADDED
print(f"Best region: C={coarse_C}, gamma={coarse_gamma}")  # ADDED
print(f"Best coarse CV balanced accuracy: {coarse_search.best_score_:.3f}")  # ADDED

# ---- Level 2: zoom in with a finer grid around the coarse best point ----  # ADDED
def zoom_bounds(value, grid_values):  # ADDED
    """Return the neighbors of `value` in the sorted coarse grid, to  # ADDED
    bound the finer grid (falls back to value/10, value*10 at the edges)."""  # ADDED
    sorted_vals = sorted(grid_values)  # ADDED
    idx = sorted_vals.index(value)  # ADDED
    lo = sorted_vals[idx - 1] if idx > 0 else value / 10  # ADDED
    hi = sorted_vals[idx + 1] if idx < len(sorted_vals) - 1 else value * 10  # ADDED
    return lo, hi  # ADDED

C_lo, C_hi = zoom_bounds(coarse_C, coarse_grid["clf__C"])  # ADDED
gamma_lo, gamma_hi = zoom_bounds(coarse_gamma, coarse_grid["clf__gamma"])  # ADDED

fine_grid = {  # ADDED
    "clf__C": np.linspace(C_lo, C_hi, 5).tolist(),  # ADDED
    "clf__gamma": np.geomspace(gamma_lo, gamma_hi, 5).tolist()  # log-spaced since gamma spans orders of magnitude  # ADDED
}  # ADDED

grid_search = GridSearchCV(  # CHANGED: was `GridSearchCV(SVC(...), param_grid=param_grid, ...)` — now uses `pipe_svm` + `fine_grid`
    pipe_svm, param_grid=fine_grid,  # CHANGED
    cv=cv_strategy, scoring="balanced_accuracy", n_jobs=-1  # CHANGED
)  # CHANGED
grid_search.fit(X_train_sub, y_train_sub)  # unchanged line, kept for clarity

print("\n--- Fine grid search results (zoomed in) ---")  # CHANGED: text updated
print(f"Best C: {grid_search.best_params_['clf__C']:.4g}")  # CHANGED: key is now `clf__C`
print(f"Best gamma: {grid_search.best_params_['clf__gamma']:.4g}")  # CHANGED: key is now `clf__gamma`
print(f"Best CV balanced accuracy: {grid_search.best_score_:.3f}")

best_C = grid_search.best_params_["clf__C"]  # CHANGED: key is now `clf__C`
best_gamma = grid_search.best_params_["clf__gamma"]  # CHANGED: key is now `clf__gamma`

# ADDED: this is the actual justification for RBF, comparing directly
# to the linear check from Section 2b -- both evaluated with the same
# CV folds and the same subsample, so the comparison is apples-to-apples.
rbf_gain = grid_search.best_score_ - linear_svm_cv_score  # ADDED
print(f"\n--- Linear vs. RBF comparison (Project.md Step 4 justification) ---")  # ADDED
print(f"Linear SVM CV balanced accuracy: {linear_svm_cv_score:.3f}")  # ADDED
print(f"RBF SVM (tuned) CV balanced accuracy: {grid_search.best_score_:.3f}")  # ADDED
if rbf_gain > 0.01:  # ADDED
    print(f"RBF improves balanced accuracy by {rbf_gain:.3f} over linear -- "  # ADDED
          f"the data is not fully linearly separable, justifying the RBF kernel.")  # ADDED
else:  # ADDED
    print(f"RBF only improves balanced accuracy by {rbf_gain:.3f} over linear -- "  # ADDED
          f"the extra complexity of the RBF kernel may not be strongly justified here.")  # ADDED
    
    
best_pipeline_probe = grid_search.best_estimator_  # ADDED
selected_mask = best_pipeline_probe.named_steps["feature_select"].get_support()  # ADDED
selected_features = X_train.columns[selected_mask].tolist()  # CHANGED: was `X_train.columns[sfs.get_support()]`
dropped_features = X_train.columns[~selected_mask].tolist()  # CHANGED: was `X_train.columns[~sfs.get_support()]`

print(f"\n--- SVM feature selection (wrapper: backward elimination) ---")
print(f"Selected features ({len(selected_features)}): {selected_features}")
print(f"Dropped features ({len(dropped_features)}): {dropped_features}")

X_train = X_train[selected_features]
X_test = X_test[selected_features]

# ADDED: scale the FINAL train/test sets too -- fit on X_train only,
# applied to X_test, mirroring exactly what 02_preprocessing.py used to
# do globally, just now scoped correctly to this final, full-data fit.
final_scaler = StandardScaler()

# FIXED: wrap the scaled numpy arrays back into DataFrames with the
# original index/columns preserved -- StandardScaler.transform() returns
# a plain ndarray, which drops .index and broke the misclassified-points
# analysis later in this script (X_test.index[...] below).
X_train = pd.DataFrame(  # FIXED: was `X_train = final_scaler.fit_transform(X_train)`
    final_scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test = pd.DataFrame(  # FIXED: was `X_test = final_scaler.transform(X_test)`
    final_scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)
# REMOVED: X_train_sub = X_train_sub[selected_features]   <- X_train_sub isn't reused after this point anymore

# =============================================================================
# SECTION 3b: LINEAR VS RBF DECISION BOUNDARY (VISUAL COMPARISON FOR SLIDES)
# =============================================================================
"""Purely for visualization -- feeds nothing into model selection or the
final results. Real feature space is 10D, so we project the SAME
subsample used for hyperparameter search (X_train_sub) down to 2 PCA
components just for this plot, then refit linear and RBF (using the
ACTUAL tuned best_C/best_gamma) on that 2D projection to draw both
decision regions side by side."""

from sklearn.decomposition import PCA

pca_2d = PCA(n_components=2, random_state=42)
X_train_sub_2d = pca_2d.fit_transform(X_train_sub)

viz_linear_svm = SVC(kernel="linear", C=1, random_state=42).fit(
    X_train_sub_2d, y_train_sub)
viz_rbf_svm = SVC(kernel="rbf", C=best_C, gamma=best_gamma, random_state=42).fit(
    X_train_sub_2d, y_train_sub)

x_min, x_max = X_train_sub_2d[:, 0].min() - 1, X_train_sub_2d[:, 0].max() + 1
y_min, y_max = X_train_sub_2d[:, 1].min() - 1, X_train_sub_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                      np.linspace(y_min, y_max, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]

y_train_sub_numeric = (y_train_sub == "g").astype(int)  # for coloring only

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, model, title in zip(
    axes,
    [viz_linear_svm, viz_rbf_svm],
    ["Linear Kernel (C=1)",
     f"RBF Kernel (C={best_C:.3g}, gamma={best_gamma:.3g})"]
):
    Z = (model.predict(grid_points) == "g").astype(int).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=class_cmap)
    ax.scatter(X_train_sub_2d[:, 0], X_train_sub_2d[:, 1],
               c=y_train_sub_numeric, cmap=class_cmap,
               edgecolors="k", s=15, linewidths=0.3)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

fig.suptitle("Linear vs. RBF SVM Decision Boundary (2D PCA projection)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "08b_svm_linear_vs_rbf_boundary.png"), dpi=150)
plt.show()
print("Plot saved: 08b_svm_linear_vs_rbf_boundary.png")


# =============================================================================
# SECTION 4: VISUALIZE THE GRID SEARCH RESULTS
# =============================================================================

results_df = pd.DataFrame(grid_search.cv_results_)
pivot_table = results_df.pivot(
    index="param_clf__gamma", columns="param_clf__C", values="mean_test_score"
)

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(pivot_table.values, cmap="viridis", aspect="auto")    # displays as heatmap

ax.set_xticks(range(len(pivot_table.columns)))  # tick mark with numbers at every column
ax.set_xticklabels(pivot_table.columns)     # replacing tese numbers with C values

ax.set_yticks(range(len(pivot_table.index)))
ax.set_yticklabels(pivot_table.index)

ax.set_xlabel("C", fontsize=11)
ax.set_ylabel("gamma", fontsize=11)
ax.set_title("SVM Grid Search: Balanced Accuracy (fine/zoomed grid)", fontsize=13)  # CHANGED: title clarifies this is the fine grid
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
"""Unlike an earlier draft's comment here, this SVM model DOES use
wrapper-based backward elimination (Section 2b+3 above), now properly
nested inside the same CV folds used for the multilevel C/gamma search
— it is NOT run on all 10 features unselected.

SVM is known to be sensitive to high-leverage points (Lecture 7: such
points can pull the decision boundary), since its boundary depends only
on the support vectors — points near the margin. A deeper pass
identifying which test points are misclassified with the largest margin
violations is left for 06_evaluation.py if time allows."""

# =============================================================================
# SECTION 10b: HIGH LEVERAGE (SUPPORT VECTORS) + MISCLASSIFICATION RANKING
# =============================================================================
"""Lecture 9: SVM's decision boundary depends ONLY on its support
vectors. That makes them exactly the "high leverage" points for this
model -- by definition, removing any non-support-vector point would
not change the boundary at all. No new statistic needed; this comes
straight from how SVM already works."""

n_support = final_svm.n_support_.sum()
pct_support = n_support / len(X_train) * 100
print(f"\n--- High leverage points for SVM: support vectors ---")
print(f"{n_support} of {len(X_train)} training points ({pct_support:.1f}%) "
      f"are support vectors -- the points the boundary actually depends on.")

"""Lecture 7: rank misclassified points by distance to the boundary.
decision_function() gives that signed distance directly for SVM."""

decision_scores = final_svm.decision_function(X_test)
misclassified_mask = (y_pred != y_test.values)

svm_margin_report = pd.DataFrame({
    "true_class": y_test.values[misclassified_mask],
    "predicted_class": y_pred[misclassified_mask],
    "margin_violation": np.abs(decision_scores[misclassified_mask]),
}, index=X_test.index[misclassified_mask]).sort_values(
    "margin_violation", ascending=False)

print(f"\n--- {misclassified_mask.sum()} misclassified test points out of {len(y_test)} ---")
print("Top 5 worst margin violations (most confidently WRONG):")
print(svm_margin_report.head(5))

# =============================================================================
# SECTION 10: SAVE RESULTS FOR 06_evaluation.py  # ADDED
# =============================================================================

os.makedirs(os.path.join("results", "model_outputs"), exist_ok=True)  # ADDED

np.savez(  # ADDED
    os.path.join("results", "model_outputs", "svm_results.npz"),
    fpr=fpr,
    tpr=tpr,
    auroc=auroc,
    sensitivity=sensitivity,
    specificity=specificity,
    selected_features=np.array(selected_features, dtype=object),
    misclassified_indices=svm_margin_report.index.to_numpy(),  # add this line
)  # ADDED

print("Results saved: results/model_outputs/svm_results.npz")  # ADDED
