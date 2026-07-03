# =============================================================================
# DECISION BOUNDARY VISUALIZATION (2D SLICE)
# =============================================================================
"""To visually compare how kNN, SVM, and Random Forest draw their decision
boundaries, we select the two most important features identified by the
Random Forest (Section 7 of 05_random_forest.py) and plot each model's
boundary in that 2D feature subspace.

WHY ONLY 2 FEATURES?
Decision boundaries can only be drawn visually in 2D. With 10 features
we cannot show the full boundary, but a 2D slice through the two most
informative features gives a meaningful picture of how each model
"thinks" — matching the visualization style from Lecture 3a (scatter
plot with class labels) and Lecture 9 (linear vs tree boundaries).

IMPORTANT CAVEAT: this is a 2D projection of a 10D decision boundary.
The actual classifiers were trained on all 10 features — this plot shows
how they behave when only 2 features are considered, not their full
behavior. We state this explicitly so the visualization is not
misinterpreted as the full decision boundary."""

# Use the two top features from RF importance — update these after running
# 05_random_forest.py and seeing the actual ranking
feat1 = "fAlpha"    # expected top feature from EDA and RF importance
feat2 = "fSize"     # update to actual 2nd-ranked feature after running RF

f1_idx = list(X_train.columns).index(feat1)
f2_idx = list(X_train.columns).index(feat2)

X_train_2d = X_train[[feat1, feat2]].values
X_test_2d = X_test[[feat1, feat2]].values
y_train_arr = (y_train == "g").astype(int).values

# Retrain each model on just these 2 features
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

knn_2d = KNeighborsClassifier(n_neighbors=best_k)
svm_2d = SVC(kernel="rbf", C=best_C, gamma=best_gamma, probability=True)
rf_2d = RandomForestClassifier(n_estimators=best_n_est, max_depth=best_depth,
                                max_features="sqrt", random_state=42)

knn_2d.fit(X_train_2d, y_train_arr)
svm_2d.fit(X_train_2d, y_train_arr)
rf_2d.fit(X_train_2d, y_train_arr)

# Build a mesh grid to evaluate the boundary across the full 2D space
x_min = X_train_2d[:, 0].min() - 0.5
x_max = X_train_2d[:, 0].max() + 0.5
y_min = X_train_2d[:, 1].min() - 0.5
y_max = X_train_2d[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]

models_2d = [
    (knn_2d, "kNN", "Blues"),
    (svm_2d, "SVM (RBF)", "Greens"),
    (rf_2d, "Random Forest", "Oranges")
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sample_2d = pd.DataFrame(X_test_2d, columns=[feat1, feat2]).sample(
    n=500, random_state=42
)
y_test_arr = (y_test == "g").astype(int).values

for ax, (model, name, cmap) in zip(axes, models_2d):
    Z = model.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    # plot a sample of test points colored by true class
    for cls_val, color, label in [(1, "steelblue", "gamma (g)"),
                                   (0, "coral", "hadron (h)")]:
        mask = y_test_arr[:500] == cls_val
        ax.scatter(
            X_test_2d[:500][mask, 0],
            X_test_2d[:500][mask, 1],
            c=color, s=10, alpha=0.5, label=label
        )

    ax.set_xlabel(feat1, fontsize=11)
    ax.set_ylabel(feat2, fontsize=11)
    ax.set_title(f"{name} decision boundary\n(2D slice: {feat1} vs {feat2})", fontsize=11)
    ax.legend(fontsize=8)

plt.suptitle(
    f"Decision Boundaries in 2D Feature Subspace ({feat1} vs {feat2})",
    fontsize=13
)
plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "17_decision_boundaries_2d.png"), dpi=150)
plt.show()
print("Plot saved: 17_decision_boundaries_2d.png")

"""WHAT TO LOOK FOR:
- kNN boundary: irregular, jagged — it traces the local majority class
  around each cluster of training points (Lecture 3a).
- SVM boundary: smoother curve defined by the support vectors and the
  RBF kernel's local influence (Lecture 9).
- Random Forest boundary: staircase/rectangular regions — decision trees
  always split parallel to feature axes, so the boundary is made of
  axis-aligned boxes (Lecture 9, Section 34/48).
These visual differences directly illustrate Lecture 9's comparison:
linear/smooth classifiers vs tree-based rectangular splits. Whether the
true boundary in this subspace is linear, curved, or box-shaped tells
you which model is structurally better suited to this data."""