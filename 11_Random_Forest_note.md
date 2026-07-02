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
