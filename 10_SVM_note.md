# =============================================================================
# 04_svm.py — Advanced Classifier 1: Support Vector Machine

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

## =============================================================================
## SECTION 2: SUBSAMPLE FOR HYPERPARAMETER SEARCH (PRACTICAL NECESSITY)

"""SVM training time grows quickly with the number of samples, especially
with an RBF kernel. With ~15000 training rows and a grid search over many
(C, gamma) combinations times 5 CV folds, training on the full set would
be very slow. Lecture 10 (Step 5) explicitly allows this: "If your dataset
is huge, you might sample a smaller portion for this step so training is
fast." We use a stratified subsample ONLY for the hyperparameter search
below — the FINAL model in Section 5 is retrained on the full training
set with the best hyperparameters found here."""

## =============================================================================
## SECTION 3: HYPERPARAMETER SEARCH — C AND GAMMA (MULTILEVEL GRID SEARCH)

"""SVM with RBF kernel has two hyperparameters (Lecture 9, Sections 12-18):

- C: controls the tradeoff between a wide margin and classifying every
   training point correctly. Large C = narrow margin, tries hard to
   classify everything correctly (risk of overfitting). Small C = wider
   margin, more tolerant of misclassified points (more generalizable).

- gamma: controls how local/tight the RBF kernel's influence is. Large
   gamma = each point only strongly influences its immediate neighborhood
   (risk of overfitting). Small gamma = smoother, more global influence.

WHY THESE SPECIFIC VALUES?
Lecture 9 explicitly recommends testing values in exponential steps
(0.01, 0.1, 1, 10, 100) rather than a fine linear grid, since both C and
gamma act on a multiplicative, not additive, scale. We use a coarse
exponential grid first (multilevel grid search, Lecture 8, Method 2),
then could zoom in around the best region if needed."""


"""WHY GridSearchCV?
This performs an exhaustive search (Lecture 8, Method 1) over every
combination in param_grid — 4 x 4 = 16 combinations — using cross-
validation on the (sub-sampled) training data only, exactly as Lecture
3b/7 requires: no test data is touched during this tuning step."""

## =============================================================================
## SECTION 4: VISUALIZE THE GRID SEARCH RESULTS

"""This heatmap shows balanced accuracy across every (C, gamma) combination
tested, making it visually clear which region of the hyperparameter space
performs best — and whether the best point sits in the middle of the grid
(good) or at an edge (a sign we may need to extend the search range)."""


"""WHAT TO LOOK FOR:
The best (C, gamma) combination should not sit exactly on the edge of the
grid — if it does, the true optimum may lie outside our tested range, and
we would need to extend param_grid further in that direction."""

## =============================================================================
## SECTION 5: TRAIN FINAL MODEL ON FULL TRAINING SET, EVALUATE ON TEST SET

"""Now we retrain using the best (C, gamma) found above, but on the FULL
training set (not the subsample used only for the search) — using all
available training data for the final model, as Lecture 10 (Step 6)
recommends: "Use as much data as you can at this stage." Only after this
do we touch the test set, for the one and only honest evaluation."""

"""WHY probability=True AND predict_proba?
Same reasoning as 03_baseline_knn.py: we need a continuous score per
point (not just a hard label) to build the ROC curve and compute auROC
(Lecture 9, Section 20)."""

## =============================================================================
## SECTION 6: CONFUSION MATRIX (ABSOLUTE AND RELATIVE)

"""Same structure as 03_baseline_knn.py, so the two models can be directly
compared in 06_evaluation.py."""

## =============================================================================
## SECTION 7: SENSITIVITY, SPECIFICITY, auROC

"""Same label convention as 03_baseline_knn.py: gamma (g) is the
signal/positive class, matching the dataset's own framing."""

## =============================================================================
## SECTION 8: ROC CURVE AND REPORTING AT REQUIRED FPR THRESHOLDS

"""Same structure as 03_baseline_knn.py, using the same FPR thresholds
specified in the dataset documentation (magic04.names), so kNN and SVM
results line up directly in 06_evaluation.py."""