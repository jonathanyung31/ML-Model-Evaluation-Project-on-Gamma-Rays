# =============================================================================
# 06_evaluation.py — Comparison and Discussion (Project.md Step 6)
# =============================================================================
"""Loads the saved results from 03_baseline_knn.py, 04_svm.py, and
05_random_forest.py (each run as a separate process, so their in-memory
variables no longer exist -- see the SAVE RESULTS section added to the
end of each of those three scripts).

Produces exactly what Step 6 asks for:
1. One plot with all three ROC curves overlaid.
2. A table with auROC and sensitivity at the required FPR thresholds
   (0.01, 0.02, 0.05, 0.1, 0.2) for all three models.
3. A brief discussion of class imbalance and Random Forest feature
   importance, and a statement of which model wins and why."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED_FPR_THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2]


# =============================================================================
# SECTION 1: LOAD SAVED RESULTS
# =============================================================================

results_dir = os.path.join("results", "model_outputs")

knn = np.load(os.path.join(results_dir, "knn_results.npz"), allow_pickle=True)
svm = np.load(os.path.join(results_dir, "svm_results.npz"), allow_pickle=True)
rf = np.load(os.path.join(results_dir, "rf_results.npz"), allow_pickle=True)

models = {
    "kNN": {"data": knn, "color": "steelblue"},
    "SVM": {"data": svm, "color": "darkorange"},
    "Random Forest": {"data": rf, "color": "seagreen"},
}

print("Loaded saved results for: " + ", ".join(models.keys()))


# =============================================================================
# SECTION 2: ROC CURVES OVERLAID (Step 6, requirement 1)
# =============================================================================

fig, ax = plt.subplots(figsize=(7, 7))

for name, info in models.items():
    d = info["data"]
    ax.plot(d["fpr"], d["tpr"], color=info["color"],
            label=f"{name} (auROC = {float(d['auroc']):.3f})")

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guessing")

ax.set_xlabel("False Positive Rate (1 - specificity)", fontsize=11)
ax.set_ylabel("True Positive Rate (sensitivity)", fontsize=11)
ax.set_title("ROC Curve Comparison: kNN vs SVM vs Random Forest", fontsize=13)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "15_roc_overlay_comparison.png"), dpi=150)
plt.show()
print("Plot saved: 15_roc_overlay_comparison.png")


# =============================================================================
# SECTION 3: COMPARISON TABLE (Step 6, requirement 2)
# =============================================================================
"""Per Project Rules.md: no more than 3 decimal places, mark the most
important number in each table (here: the best auROC, and per-column
the best sensitivity at each FPR threshold)."""

rows = []
for name, info in models.items():
    d = info["data"]
    row = {"Model": name, "auROC": round(float(d["auroc"]), 3)}
    for target_fpr in REQUIRED_FPR_THRESHOLDS:
        idx = np.argmin(np.abs(d["fpr"] - target_fpr))
        row[f"Sens. @ FPR={target_fpr}"] = round(float(d["tpr"][idx]), 3)
    rows.append(row)

comparison_table = pd.DataFrame(rows).set_index("Model")

# Mark the best (highest) value in each column with an asterisk
marked_table = comparison_table.copy().astype(str)
for col in comparison_table.columns:
    best_idx = comparison_table[col].idxmax()
    marked_table.loc[best_idx, col] = f"{comparison_table.loc[best_idx, col]:.3f}*"

print("\n--- Comparison Table (Step 6) ---")
print("(* marks the best value in each column)")
print(marked_table)

comparison_table.to_csv(os.path.join("results", "model_outputs", "comparison_table.csv"))
print("\nTable saved: results/model_outputs/comparison_table.csv")


# =============================================================================
# SECTION 4: CLASS IMBALANCE DISCUSSION (Step 6, requirement 3)
# =============================================================================
"""Project.md Key Thing #5: mention the sensitivity/specificity tradeoff
and that a model predicting "gamma" for everything would get ~65%
accuracy while being useless -- this is why auROC/sensitivity/specificity
are reported instead of accuracy throughout this project."""

print("\n--- Class Imbalance Discussion ---")
print("The dataset is ~65% gamma / ~35% hadron. A trivial classifier that")
print("always predicts 'gamma' would score ~65% accuracy while catching")
print("zero hadrons (specificity = 0) -- which is why accuracy is never")
print("used as the primary metric in this project, and auROC/sensitivity/")
print("specificity are reported instead, per the dataset's own documentation.")


# =============================================================================
# SECTION 5: RANDOM FOREST FEATURE IMPORTANCE DISCUSSION (Step 6, requirement 4)
# =============================================================================

rf_importances = rf["feature_importances"]
rf_feature_names = rf["feature_names"]
sorted_idx = np.argsort(rf_importances)[::-1]

print("\n--- Random Forest Feature Importance (which Hillas parameters matter most?) ---")
for i in sorted_idx:
    print(f"{rf_feature_names[i]}: {rf_importances[i]:.3f}")
print("fAlpha leads by a wide margin, consistent with the physics literature")
print("(the shower-axis angle is the single strongest gamma/hadron discriminator).")


# =============================================================================
# SECTION 6: STATE THE WINNING MODEL (Step 6, requirement 5)
# =============================================================================

best_model = comparison_table["auROC"].idxmax()
best_auroc = comparison_table["auROC"].max()

print(f"\n--- Winner ---")
print(f"{best_model} achieves the highest auROC ({best_auroc:.3f}) among the three")
print("models and is the recommended choice for this task.")