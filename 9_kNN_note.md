# =============================================================================
# 03_baseline_knn.py — Base Experiment: k-Nearest Neighbors

A baseline model means:
“Use the dataset in a simple, standard form and train the model without tailoring the feature space to that specific algorithm.”

"""WHAT IS THIS FILE?
This is the third step of the project: training our baseline classifier,
kNN. kNN is our base model because it is simple, parameter-free (no
learned weights), and easy to explain (Lecture 3a) — it gives us a
starting point that any "advanced" classifier (SVM, Random Forest) needs
to beat to justify its added complexity.

WHY DOES KNN NEED NORMALIZED DATA?
kNN classifies a point by finding its closest neighbors using distance
(Lecture 3a). If features are on different scales, the largest-scale
feature would dominate every distance calculation. This is why we use the
already-normalized data produced by 02_preprocessing.py, not the raw data."""

"""NOTE: .squeeze() converts a single-column DataFrame (what to_csv/read_csv
gives us back for y_train/y_test) into a plain pandas Series, which is the
format sklearn expects for labels."""


## SECTION 2: CHOOSE k VALUES TO TEST (HYPERPARAMETER TUNING)

"""k is kNN's only real hyperparameter (Lecture 3a/3b) — it is NOT learned
from the data, we have to choose it ourselves. We test several values and
use cross-validation on the TRAINING set only to pick the best one,
exactly as Lecture 3b/7 describes: the test set must not be touched for
any decision-making, only for the final reported result.

WHY THESE SPECIFIC VALUES?
We try a spread from very small (k=1, most flexible, most overfitting
risk) to larger (k=20, smoother, more underfitting risk), matching the
overfitting-vs-underfitting tradeoff shown in Lecture 3b, Section 1-2."""

"""WHY CROSS-VALIDATION INSTEAD OF A SINGLE VALIDATION SPLIT?
Lecture 7 explains that with a single random split, the result can be
unstable — depending on which points happen to land in the validation
set. k-fold cross-validation (we use 5 folds here, a typical choice per
Lecture 7) averages over several splits, giving a more reliable estimate
of how each k value will generalize."""

LLN means that if you average results over many repeated samples, the result becomes more stable and closer to the “true” performance.

LLN explains why averaging over 5 CV folds gives a more reliable estimate than a single train/validation split — and the std across folds tells us how far we still are from that "stable, true" value with our current amount of data. A small std means our balanced-accuracy estimate for a given k is trustworthy; a large std means we'd want more data or more folds before fully trusting it.

"""WHY STRATIFIED K-FOLD SPECIFICALLY?
Just like our train/tes t split, each fold must also preserve the
~65/35 class ratio - otherwise some folds could end up with a very
different class balance than others, making the cross-validation
results noisier than necessary."""

## Different Cross-validation techiniques:
  Why not fewer folds? 2-fold or 3-fold: - too much data variation per split, unstable estimates, high randomness

  Why not more folds (like 10 or LOOCV)? 10-fold: - slightly better estimate, but 2× slower than 5-fold

  Leave-One-Out (LOOCV): - almost unbiased, but extremely expensive, also higher variance in some cases

  5-fold cross-validation is a practical trade-off between reliability and computational cost. Why 5-fold is common in practice - In real machine learning: 5-fold = standard default in many pipelines, works well for medium datasets, good balance for hyperparameter tuning (like kNN), That’s why your lecture says: “a typical choice per Lecture 7”

## Overfitting vs. underfitting — why does k control this?

k=1 (overfitting risk): each prediction is based on just the single nearest neighbor. The model can carve an extremely jagged, twisty decision boundary that hugs every individual training point — including noise. It will look great on training data (it can even memorize it) but generalize poorly to new points.

k=25 (underfitting risk): each prediction averages the vote of 25 neighbors, smoothing things out heavily. If the true boundary between gamma and hadron has any fine detail, a large k will blur right past it — the model becomes too rigid/smooth to capture real structure.

This is the exact U-shaped tradeoff your own Lecture 3b notes describe: training error keeps dropping as k shrinks (more flexible, more overfitting), while test/validation error drops, hits a sweet spot, then rises again at both extremes.

Why specifically k_values = list(range(1, 31))?
it's a reasonable spread covering the overfitting end (1 - 4), a plausible middle range (5-20), and a clearly smooth/underfitting end (20-31), so the resulting plot shows the full U-shape rather than just one side of it. There's no rule in your slides dictating exact values to test — Lecture 3b's own example just used specific values (10, 1, 100) to illustrate the concept, not to prescribe a testing grid.

---

shuffle=True -> This means: Before splitting, the data is randomly shuffled. Why?
prevents ordered data from biasing folds, ensures each fold is more representative

### Normal vs. Balanced Accuracy

Normal accuracy would do this: Count total correct predictions, Let the majority class dominate the score. So if most samples are gamma or hadron, the model could: ignore the minority class, still look “good” on paper. Which is NOT good!

Balanced accuracy fixes this by: Giving equal importance to both classes, It computes:
accuracy on gamma, accuracy on hadron and then averages them. Meaning, this ensures: "Did you perform well on BOTH gamma and hadron?" not just the dominant one.

## =============================================================================
## SECTION 3: PLOT VALIDATION ERROR vs k

"""This plot directly recreates the kind of overfitting/underfitting curve
shown in Lecture 3b, Section 1 (kNN with simulated data): small k overfits
(jagged, memorizes training data), large k underfits (too smooth). We
expect to see a U-shape or similar — error high at very small k, dropping
to a minimum, then rising again at very large k."""

"""WHAT TO LOOK FOR:
The k with the lowest cross-validated error is our chosen hyperparameter.
This decision was made using ONLY the training set (via cross-validation)
— the test set has not been touched yet."""

## =============================================================================
## SECTION 4: TRAIN FINAL MODEL WITH BEST k, EVALUATE ON TEST SET

This uses all available training data, not folds anymore

"""Now, and only now, do we touch the test set. This is the one and only
evaluation it gets (Lecture 10 — "data snooping" warning): we train the
final model (with the k chosen above) on the full training set, and
report its performance on the test set as our honest, final result."""

"""WHY predict_proba AND NOT JUST predict?
predict() gives a hard class label (g or h). But to compute ROC curves
and auROC (Lecture 3b/5b), we need a continuous probability/score per
point, so we can sweep a threshold across all possible values, exactly
as Lecture 5b's ROC curve section describes."""

## =============================================================================
## SECTION 5: CONFUSION MATRIX (ABSOLUTE AND RELATIVE)

"""Lecture 3b explicitly recommends showing BOTH the absolute confusion
matrix (raw counts) and the relative confusion matrix (each row divided
by its true class total) — the relative version makes the per-class
classification rate easy to read directly, which matters here since our
classes are imbalanced."""

Absolute Confusion Matrix shows actual counts of each prediction:
        predicted
          g    h
true g   900  100
true h   80   720

Interpretation:
900 gamma correctly classified
100 gamma missed
80 hadron misclassified as gamma
720 hadron correctly classified

Relative Confusion Matrix shows normalized counts of each prediction:
this means this is the formula `value = cell / total counts in row` 

        predicted
          g     h
true g   0.90  0.10
true h   0.10  0.90

## =============================================================================
## SECTION 6: SENSITIVITY, SPECIFICITY, auROC

"""WHY NOT JUST ACCURACY?
Both the dataset documentation and Lecture 3b are explicit: with ~65/35
imbalanced classes, accuracy can look good while the model is still bad
at catching the minority class. We compute sensitivity and specificity
directly from the confusion matrix (Lecture 3b, Section 10), and auROC
as our single most meaningful summary number (Lecture 3b/5b)."""

The auROC uses sensitivity (another name for it: recall or TPR (True Positive Rate)),
and False Positive Rate (FPR) which is calculated by 1 - specificity(TNR).
It is done like this because we only care to detect as many correct gamma while trying to avoid as much as possible detecting hadron as false gammas.

"""WHAT TO LOOK FOR:
auROC close to 1.0 = strong separation between classes (Lecture 3b/5b).
auROC close to 0.5 = no better than random guessing.
We expect a reasonably high auROC here given fAlpha's strong individual
separation found in 01_eda.py."""

## =============================================================================
## SECTION 7: ROC CURVE AND REPORTING AT REQUIRED FPR THRESHOLDS

"""The dataset documentation (magic04.names) explicitly specifies the FPR
thresholds at which sensitivity should be compared between classifiers:
0.01, 0.02, 0.05, 0.1, 0.2. We compute the ROC curve and report
sensitivity at exactly these points, so this baseline can be directly
compared against SVM and Random Forest later in 06_evaluation.py."""

Top-left point = ideal but usually theoretical best case
AUROC = overall quality across all possible thresholds
AUROC does NOT come from one best-performing threshold

required_fpr_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]
A list of specific false alarm levels you want to check your model at.
Each one is a precentage of FP. They are not random — they are:
predefined evaluation points from your dataset instructions.

So instead of just one score (like AUROC), you evaluate:
“How does the model perform under different levels of strictness?”

You loop over them and ask:
“At each FPR level, how much signal (TPR) do I get?”

🔴 Very strict region: 0.01, 0.02, almost no false alarms allowed, used when false positives are very expensive
🟠 Medium strictness: 0.05, common “engineering trade-off zone”
🟡 Relaxed detection: 0.1, 0.2, allow more false alarms to catch more true signals

Standard practice in ROC-based research - These values are very common “benchmark points” in classification papers. “standard checkpoints along the ROC curve that represent meaningful operating conditions”

AUROC - “How good is the model overall at separating g vs h?”
ROC + thresholds - “What happens if I make the model strict or lenient?”

"""WHAT TO LOOK FOR:
This table is the exact comparison structure the dataset's original
authors intended (magic04.names, Section 9). We will reuse this same
table format for SVM and Random Forest, so all three models can be lined
up side-by-side in 06_evaluation.py."""

## =============================================================================
## SECTION 8: NOTE ON FEATURE SELECTION AND HIGH LEVERAGE / OUTLIERS

"""Per Lecture 3b (mistake #2) and Lecture 7, feature selection and any
outlier/high-leverage analysis must be model-specific, not done once in
preprocessing. For this baseline kNN model, we deliberately use all 10
features without selection, to establish a true baseline before any
feature engineering. Outlier/high-leverage analysis specific to kNN's
predictions (Lecture 7: misclassified points farthest from the decision
boundary) is left for a deeper pass if time allows, but is not required
for the baseline to be valid."""
