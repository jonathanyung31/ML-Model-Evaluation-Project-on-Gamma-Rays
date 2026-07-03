# =============================================================================
# 05_random_forest.py — Advanced Classifier 2: Random Forest

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


## =============================================================================
## SECTION 1: IMPORTS AND LOAD PREPROCESSED DATA

kNN and SVM don't produce any output that requires knowing which feature is which by name — kNN just votes among neighbors, SVM just computes distances to the hyperplane, and neither gives you a per-feature score or ranking to label. So there was nothing to label in those files, and feature_names was never needed.

The line needs to come before any numpy conversion because once you do operations like .values on a DataFrame, the column names are lost — you're left with a plain array of numbers with no labels attached. So it's stored early as a precaution:

## =============================================================================
## SECTION 2: HYPERPARAMETERS OF RANDOM FOREST

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

## =============================================================================
## SECTION 3: HYPERPARAMETER TUNING VIA CROSS-VALIDATION

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


"""WHY STRATIFIED K-FOLD AGAIN?
Same reasoning as 03_baseline_knn.py: each fold must preserve the
~65/35 class ratio so cross-validation results are not distorted by
folds with an unrepresentative class balance."""

"""WHY BALANCED ACCURACY FOR TUNING?
Same justification as 03_baseline_knn.py: it directly addresses the
class imbalance found in 01_eda.py. Final headline comparison between
models still uses auROC — balanced accuracy here is only for selecting
hyperparameters during cross-validation, not the final reported metric.
The std across folds is also printed to check stability of the estimate,
connecting to Lecture 7's point that a single number without spread
information can be misleading."""


## =============================================================================
## SECTION 4: VISUALIZE HYPERPARAMETER SEARCH RESULTS

"""We show a heatmap of balanced accuracy across all (n_estimators,
max_depth) combinations, exactly as we did for SVM's (C, gamma) grid in
04_svm.py. This makes the best region visually obvious and lets us check
whether the best combination sits at an edge of the grid (which would
suggest we should extend the search range)."""

WHAT TO LOOK FOR:
We check whether performance changes significantly across the hyperparameter grid. In our results, performance quickly increases and then plateaus, with similar balanced accuracy for n_estimators ≥ 100. This indicates that adding more trees beyond this range does not meaningfully improve performance, so the model has already stabilized.

For max_depth, shallow trees (e.g. 5) underperform clearly, while deeper trees (10, 20, and None) perform similarly. This suggests that the model benefits from deeper trees up to a point, but full growth (max_depth=None) does not significantly outperform moderately deep trees, meaning there is no strong evidence that further tuning of depth would yield large gains.

Overall, no strong edge effect is observed in the grid, so the current hyperparameter range is sufficient and does not need to be extended.

## =============================================================================
## SECTION 5: TRAIN FINAL MODEL ON FULL TRAINING SET, EVALUATE ON TEST SET

"""Same principle as 04_svm.py Section 5: retrain on the FULL training
set (not any subsample) with the best hyperparameters found above, then
touch the test set exactly once for the honest, final evaluation.
(Lecture 10 Step 6: "use as much data as you can at this stage.")"""
