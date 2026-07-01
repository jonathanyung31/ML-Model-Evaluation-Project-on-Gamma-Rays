# 01_eda.py — Exploratory Data Analysis (EDA)

""" WHAT IS THIS FILE?
 This is the first step of any machine learning project: understanding your
 data before touching any algorithm. We look at the structure, distributions,
 class balance, and relationships between features.

 WHY DO WE DO THIS FIRST?
 Every decision we make later (which model to use, how to preprocess, which
 metric to evaluate with) must be justified by what we find here. The
 professor grades reasoning, not just results — this file builds that reasoning.
 ============================================================================="""

## SECTION 0: PROBLEM REASONING (Lecture 3a)

""" Before touching any algorithm, the course asks us to answer:
 1. What classes/data are we trying to predict? -> binary: gamma (g) vs hadron (h)
 2. Is a connection between data and result visible with simple methods?
    -> we check this below with histograms, summary stats, and scatter plots.
 3. Is there hope for success at all?
    -> if classes show separation in simple plots, yes; this EDA answers that."""

## SECTION 1: IMPORTS

"""We import the libraries we need. Think of libraries as toolboxes —
each one gives us ready-made tools so we don't have to write everything
from scratch."""

## SECTION 2: LOAD THE DATA

The dataset is a plain text file with comma-separated values but NO header
row (no column names written in the file itself). We have to supply the
column names manually, which we know from reading magic04.names.

Define the column names exactly as described in magic04.names

    "fLength",      # major axis of ellipse [mm]  | length of the shower shape (long side)
    "fWidth",       # minor axis of ellipse [mm]  | width of the shower shape (short side)
    "fSize",        # 10-log of sum of content of all pixels [#phot]  | overall          brightness / total signal strength
    "fConc",        # ratio of sum of two highest pixels over fSize  | how concentrated the brightest pixels are
    "fConc1",       # ratio of highest pixel over fSize  | how dominant the single brightest pixel is
    "fAsym",        # distance from highest pixel to center, projected onto major axis [mm]  | how off-center the brightness is
    "fM3Long",      # 3rd root of third moment along major axis [mm]  | skewness / asymmetry along main direction
    "fM3Trans",     # 3rd root of third moment along minor axis [mm]  | skewness across the main direction
    "fAlpha",       # angle of major axis with vector to origin [deg]  | orientation angle of the shape
    "fDist",        # distance from origin to center of ellipse [mm]  | how far the event is from the detector center
    "class"         # target label: 'g' = gamma (signal), 'h' = hadron (background)  | classification label (signal vs background)

Load the data from the file. Using os.path.join() to adapt to other os
pd.read_csv works for any comma-separated file, not just .csv files

Since we only read from magic04.data and never write back to it, the
original file is never at risk — but we make this explicit by working
with an in-memory copy throughout.

## SECTION 3: BASIC STRUCTURE
 Before anything else, we check the basic facts about our dataset:
 how many rows, how many columns, what types of data, any missing values?

### WHAT TO LOOK FOR:
We expect 19020 rows and 11 columns (10 features + 1 class label).
If the numbers are wrong, the file didn't load correctly.

### WHAT TO LOOK FOR:
All feature columns should be float64 (decimal numbers).
The 'class' column should be an object (text: 'g' or 'h'). in pandas, “object” is the default 
data type used to store strings when a column doesn’t fit into a numeric type
If something looks wrong here, the column names may be misaligned or data not read correctly.

### WHAT TO LOOK FOR:
 This just gives us a quick visual check that the data was loaded
 correctly and columns are aligned with their names.

### WHAT TO LOOK FOR:
 We expect 0 missing values in every column.
 The UCI page confirms there are no missing values in this dataset.
 If we see any non-zero numbers here, something went wrong with loading.

### WHAT TO LOOK FOR:
 All numeric columns should have reasonable min/max values.
 Example: no negative values where they don't make sense.
 Extremely large or tiny values may indicate scaling issues or corrupted data.

### WHAT TO LOOK FOR:
 Should only contain two values: 'g' and 'h'.
 If you see anything else, it means data is corrupted or misread.

### WHAT TO LOOK FOR:
We expect 0 duplicate rows, or at least very few.
If we find a meaningful number of duplicates, we should remove the extra ones

### WHAT TO LOOK FOR:
We expect all values to be 0.
If any are non-zero, it means missing data is hidden as text instead of NaN.
These must be converted to NaN before analysis.


# SECTION 4: CLASS DISTRIBUTION

""" Our target variable has two classes: 'g' (gamma, signal) and 'h' (hadron,
 background). We need to check how many samples belong to each class.

    WHY DOES THIS MATTER?
    If one class has far more samples than the other (called "class imbalance"),
    then a naive model can cheat — it can predict the majority class every time
    and still get high accuracy.
    The MAGIC dataset description explicitly warns us about this, and even
    specifies exact thresholds for the proper evaluation metric (ROC curve),
    which we will apply later when comparing models in 06_evaluation.py.

    IMPORTANT NOTE FROM THE DATASET DOCUMENTATION (magic04.names):
    This data is Monte Carlo generated, and the documentation explicitly
    states: "For technical reasons, the number of h events is underestimated.
    In the real data, the h class represents the majority of the events."
    So the ~65/35 gamma/hadron split we see here is an artifact of how this
    simulated dataset was built, NOT a reflection of the true real-world class
    balance, where hadron background events actually dominate."""

"""WHAT WE FOUND:
g (gamma/signal):   ~12332 samples (~64.8%)
h (hadron/background): ~6688 samples (~35.2%)
This is a moderately imbalanced dataset in this sample — though as noted
above, the imbalance direction is actually reversed in real-world data.
Either way, we CANNOT use accuracy as our main evaluation metric here.
We will use ROC-AUC, sensitivity, and specificity instead.
The dataset documentation also specifies exact false-positive-rate
thresholds (0.01, 0.02, 0.05, 0.1, 0.2) at which sensitivity should be
reported for a proper comparison between classifiers — we apply this in
06_evaluation.py, once we have trained models to evaluate."""


"""IMPORTANT FOR PRESENTATION: the "real data has h as majority" fact is a 
great one-liner for your 
Motivation slide (Lecture-required: "How the dataset has been generated 
must be answered!") — it shows you read the documentation closely and understand 
a real limitation of working with simulated data, which is exactly the kind of 
critical reasoning that gets rewarded."""

# SECTION 5: SUMMARY STATISTICS PER CLASS

""" We compute basic statistics (mean, std, min, max) for each feature,
 separated by class. This tells us whether the two classes look different
 in terms of their feature values — which is a prerequisite for any
 classifier to work at all.
 Better data beats a better algorithm."""

""" WHAT TO LOOK FOR:
 Compare the means between the two classes for each feature.
 If a feature has very different means for g vs h, it is likely useful
 for discrimination (the ability to distinguish between classes). 
 If the means are nearly identical, the feature may not help much.
 fAlpha in particular is known from physics to be highly discriminative:
 gamma showers point toward the telescope source, hadronic showers don't."""


# SECTION 6: FEATURE DISTRIBUTIONS PER CLASS

""" We plot a histogram for each feature, with the two classes overlaid.
 This gives us a visual version of the summary statistics above.
 Overlaying helps you quickly see: separation between classes, whether a 
 feature is good for classification and how much the distributions overlap

 WHY HISTOGRAMS?
 They show the full shape of the distribution, not just the mean.
 Two features can have different means but completely overlapping
 distributions — in which case a classifier will struggle to use them.
 Conversely, even slight separation in the distributions is useful."""

We are trying to see if there are an differentiating feature values between the classes
So we can know which features are more important for classification than others

""" WHAT WE FOUND (compared to expectations):
 - fAlpha: confirmed strong separation as expected — gamma peaks sharply
   near 0 degrees, hadron is much flatter and more spread out.
 - fLength, fWidth: mild separation visible, less than fAlpha.
 - fSize: separation is weaker than initially expected — the two
   distributions largely overlap.
 - fAsym, fM3Long, fM3Trans: distributions overlap heavily for both
   classes — these look like weak discriminators on their own."""

# SECTION 7: CORRELATION MATRIX

""" A correlation matrix shows how strongly each pair of features is linearly
 related to each other. Values close to 1 or -1 = strong relationship.
 Values close to 0 = little to no linear relationship.

 WHY DOES THIS MATTER?
 If two features are highly correlated (e.g. 0.95), they carry almost
 the same information. Including both doesn't help much and can cause
 problems - 2 features being redundent.
 Redundancy = duplicate or overlapping information that isn't needed. 
 This informs feature selection later.

 IMPORTANT WARNING — correlation ≠ causation:
 Just because two features move together doesn't mean one causes the other.
 Just because shark attack and ice cream purchses rise on the same dates
 don't mean that either of them cause the other
 We use correlation to spot redundancy, not to causation or meaning."""

""" 
Positive correlation (e.g., fLength vs fWidth = 0.77): as one feature goes up, 
the other tends to go up too. Bigger length → bigger width.
Negative correlation (e.g., fSize vs fConc = −0.85): as one feature goes up, 
the other tends to go down. Bigger fSize → smaller fConc.

WHAT WE FOUND (Section 7 — correlation matrix):
 - fConc and fConc1: r = 0.98, confirming our expectation of strong
   redundancy (both measure pixel concentration ratios).

 - fLength and fWidth: r = 0.77, correlated as expected, but weaker
   than fConc/fConc1.

 - fSize and fConc/fConc1: r = -0.85 / -0.81 — a strong relationship we
   did not anticipate beforehand, stronger even than fLength/fWidth.

 - fAsym, fM3Long, fM3Trans: all near 0 with every other feature,
   as expected — they capture independent information."""

# SECTION 8: SCATTER PLOTS FOR KEY FEATURE PAIRS

""" We pick a few of the most physically meaningful feature pairs and plot
 them as scatter plots, coloring points by class. This is the clearest
 visual test of whether classes are separable.

 We can't plot all 45 possible pairs (10 features = 45 combinations),
 so we select the most informative ones based on the histogram findings.

 We use a sample of the data to keep the plot readable

 Pairs chosen for two different reasons:

 - (fAlpha, fLength), (fAlpha, fWidth), (fAlpha, fSize) → came from the 
   histograms (fAlpha was the strongest individual separator; testing it against others).

 - (fWidth, fLength), (fConc, fSize), (fConc, fConc1) → came from the correlation matrix 
   (your three highest correlation values: 0.77, -0.85, 0.98)."""


""" 
WHAT WE FOUND (Section 8 — scatter plots):

 - fAlpha vs fLength/fWidth: gamma clusters tightly in the low-fAlpha,
   low-value corner; hadron spreads out across both axes. Confirms
   fAlpha's individual separation generalizes well in 2D.

 - fAlpha vs fSize: weaker separation, consistent with fSize's poor
   individual separation in the histograms.

 - fWidth vs fLength (r=0.77): visualizes the correlation as a diagonal
   trend, but ALSO shows some separability — gamma is denser in the
   bottom-left. Correlation does not rule out separability here.

 - fConc vs fSize (r=-0.85) and fConc vs fConc1 (r=0.98): both show
   classes almost fully intermixed along the correlation curve/line — no
   visible separability. The strongest correlations in our data
   correspond to the weakest separability among the pairs we checked.
   
   We can see from the scatter plots that levels of correlation and the histograms
   expectations do lead to same level of seperability
"""

# Presentation Worthy Words of Understandings:

A few closing thoughts, pulling together what's actually presentation-ready from this EDA work:
What you can confidently put on slides, with course backing:

Class distribution — ~65/35 gamma/hadron split. This single slide does double duty: 
shows the data, and sets up your justification for why accuracy is wrong and ROC/auROC 
is right (Lecture 3b). Strong, simple, defensible.

Feature separation story — fAlpha is your headline finding, backed by physics 
reasoning (shower pointing toward source) AND visual confirmation in both 
histograms and scatter plots. This is your strongest "we understand the domain" 
moment — lead with it.

Correlation matrix + the redundancy/separability distinction — fConc/fConc1 (0.98) 
and fConc/fSize (−0.85) being highly correlated but showing no class separation, 
versus fWidth/fLength (0.77) showing some separation despite being correlated too — 
that's a genuinely interesting, data-backed nuance. It shows you're not just running 
plots, you're interpreting them critically. Worth a sentence in your "Data" or 
"Discussion" section.

Honest framing throughout — several of your "what we found" sections diverge from 
initial expectations (fSize weaker than predicted, fAsym/fM3Long/fM3Trans nearly 
indistinguishable between classes). Don't hide this — say it. The professor 
explicitly grades reasoning over results, and showing "we expected X, found Y, and 
here's what that tells us" is exactly the kind of self-correcting process that's rewarded.

What stays out of the presentation:

The Cohen's d/separability score work — dropped, since it's not slide-traceable.
IQR-based leverage checks — also dropped for the same reason; if you want to 
mention high-leverage points or outliers at all in your final report, keep it at the 
level your slides actually support (visual inspection, or true outlier/residual 
analysis later once you have a trained model).

One thing to keep in mind going forward: this EDA only tells you about individual and 
paired feature behavior. None of it yet tells you how a real classifier will perform — 
that's the entire point of Steps 3-5 next (kNN baseline, SVM, Random Forest). It's 
worth saying explicitly in your presentation that EDA gave you hypotheses (fAlpha 
matters most, fConc/fConc1 likely redundant) which your modeling results will need 
to either confirm or correct — e.g., Random Forest's feature importance output later 
is a good place to circle back and check whether fAlpha really dominates, or whether 
the classifier finds value in features your EDA underestimated."""

---
---
---

# =============================================================================
# 02_preprocessing.py — Data Preparation

"""WHAT IS THIS FILE?
This is the second step of the project: preparing the data so it is ready
to be fed into a classifier. This includes cleaning, splitting into
train/test sets, and normalizing features.

WHY DO WE DO THIS BEFORE MODELING?
Every classifier we use later (kNN, SVM, Random Forest) depends on this
file producing a train/test split and normalization that is done honestly —
meaning no information from the test set leaks into training"""

Since we only read from magic04.data and never write back to it, the
original file is never at risk — but we make this explicit by working
with an in-memory copy throughout.

## SECTION 2: MISSING VALUES (NO ACTION NEEDED) & CHECK FOR DUPLICATE ROWS (NO REMOVAL)

"""01_eda.py confirmed 0 missing values across every column, and 0 hidden
missing-value placeholders (e.g. "?", "NA"). Lecture 10's data preparation
checklist includes handling missing values (fill or delete) as a required
step — we explicitly note here that this step requires no action for this
dataset, rather than silently skipping it."""

"""01_eda.py already checked for duplicate rows. We check again here just
to document the count here as well before splitting.

WHY NOT JUST REMOVE THEM?
Removing a data point is only
acceptable if you are certain it's an error AND can explain why. "My
results look better without it" is never a valid reason. We have no proof
that duplicate rows here are logging errors rather than genuinely repeated
or very similar physical events from the simulation that generated this
dataset. So we keep them, and only revisit this decision if we later find
concrete evidence they are mistakes."""

"""duplicate rows are far more plausible to be real, independent simulated events 
that happen to land on very similar parameter values — not logging errors."""

## SECTION 3: TRAIN / TEST SPLIT
"""We split the data into a training set (80%) and a test set (20%).

WHY 80/20?
With 19020 samples, an 80/20 split still leaves a large test set (~3800
samples) — large enough to give a stable, reliable estimate of performance,
while keeping most of the data available for training.

WHY STRATIFIED?
01_eda.py found the classes are imbalanced (~65% gamma, ~35% hadron). A
plain random split could, by chance, end up with a different class ratio
in the test set than in the training set. Stratifying the split forces
both the training set and the test set to preserve the same ~65/35 ratio
found in the full dataset. This matches Lecture 3b's warning about hidden
connections and representativeness between train and test data."""


"""WHAT TO LOOK FOR:
Both the training set and the test set should show around the same
class proportions as the full dataset (~65% gamma, ~35% hadron). If the
proportions differ by a lot, the split did not stratify correctly."""

"""IMPORTANT RULE GOING FORWARD (Lecture 10 — "data snooping"):
From this point on, X_test and y_test are LOCKED AWAY. We do not look at
them again, plot them, or use them to make any decision until the very
final evaluation in 06_evaluation.py. Any hyperparameter tuning in
03/04/05 will use cross-validation on X_train/y_train only."""

## SECTION 4: NORMALIZE FEATURES

"""kNN and SVM both rely on distances between points (Lecture 3a kNN,
Lecture 9 SVM kernels). If features are on very different scales (e.g.
fAlpha ranges roughly 0-90, fAsym ranges into the hundreds), the
larger-scale feature would dominate any distance calculation, making the
smaller-scale features almost meaningless. Normalization fixes this by
rescaling every feature to have mean 0 and standard deviation 1
(Lecture 3b, Section 7)."""

"""CRITICAL TRAP (Lecture 3b, Section 7 and Lecture 7):
The mean and standard deviation used above are computed from X_train
ONLY. We then reuse those exact same values to normalize X_test. We never
compute statistics on the combined train+test data, and we never compute
separate statistics for the test set. If we computed the test set's
own statistics, we would be using information from the test set to decide
how the data is transformed — even before any model sees it. That's a
form of "peeking" at the test set. It would make the test set no longer
a fair, untouched measure of real-world performance, since the
preprocessing itself was shaped by data the model is supposed to have
never seen."""

"""WHY EXACTLY 0 AND 1 FOR THE TRAINING SET?
This is NOT because normalization makes the data normally distributed —
it does not change the shape of a feature's distribution at all (a skewed
feature stays skewed). It is pure arithmetic: subtracting the mean from
every value always shifts the average to exactly 0, and dividing every
value by the standard deviation always rescales the spread to exactly 1.
Since mean_train/std_train were computed directly FROM the training data,
applying this formula to that same data gives EXACTLY 0/1 (any tiny
deviation shown is only floating-point rounding, not a real difference).

WHY NOT EXACTLY 0/1 FOR THE TEST SET?
We reused the TRAINING set's mean/std on the test set, not the test set's
own. The test set is a different random sample with its own slightly
different actual mean/std, so after applying the training-derived
formula, its mean/std will be CLOSE to 0/1 but not exact — this is
expected and correct. If the test set's mean/std came out exactly 0/1,
that would mean we had instead computed its own statistics and normalized
with those — exactly the leak we are trying to avoid.

RANDOM FOREST:
Decision trees split one feature at a time using a threshold (e.g.
"fAlpha < 17?"), not a distance calculation across all features at once.
This means normalization has no real effect on Random Forest's splits or
performance. We still apply the same normalized data to it for pipeline
consistency across all three models, but we will note in 05_random_forest.py
that this step was not strictly necessary for that specific model."""

## SECTION 5: SAVE THE PREPARED DATA

"""We save the split and normalized data to disk so that 03_baseline_knn.py,
04_svm.py, and 05_random_forest.py can all load the exact same prepared
data, rather than each repeating this preprocessing independently. This
also guarantees every model is evaluated on an identical train/test split,
making later comparisons in 06_evaluation.py fair."""

"""
Why normalize — and do your specific models need it?

- kNN: needs it. Lecture 3b is explicit — kNN relies on distance, and if one 
feature ranges 0–10000 and another 0–1, the larger one dominates the distance 
calculation entirely, making the smaller feature meaningless. You have exactly this 
kind of scale mismatch (fAlpha ~0–90, fAsym into the hundreds).

- SVM: needs it. Lecture 9's hyperplane math and especially the RBF 
kernel (exp(-γ·distance²)) are distance-based the same way kNN is — unscaled 
features would distort the margin and kernel calculations.

- Random Forest: does NOT need it. Decision trees (Lecture 9) split by asking 
"is feature X < threshold?" one feature at a time — there's no distance 
calculation and no comparison across features happening simultaneously. 
A tree splitting on fAlpha at value 17 works identically whether fAlpha is 
in its raw units or normalized, since the split point and resulting groups 
are the same either way.

So normalization is necessary for your base classifier and one of your advanced ones, 
but technically unnecessary for Random Forest. Worth stating this explicitly 
somewhere — it's exactly the kind of "why did you do this" reasoning the professor 
grades. You can still apply the same normalized data to Random Forest (it won't hurt 
accuracy), but you should be ready to say "we know RF doesn't strictly need this, we 
used the same prepared data for pipeline consistency across models."
"""

"""
This reasoning of why these 3 models where chosen over others belongs in your 
presentation (the "Method" section, per the Scientific Presentations slide deck:
— "Explain how you constructed the machine learning evaluation. This includes: 
Model and feature selection...").

A reasonable justification structure, grounded in your own course material:

- kNN (base): simple, parameter-free, easy to explain, the natural starting baseline 
(Lecture 3a explicitly frames it as the "first classifier that actually works").

- SVM (advanced 1): continuous, fairly clean features (Hillas parameters) — 
exactly the kind of data your notes say SVM with RBF handles well; also produces 
ROC curves naturally, which the dataset explicitly requires (Lecture 9, Section 20).

- Random Forest (advanced 2): handles the class imbalance more gracefully than 
purely distance-based methods, gives feature importance for free (useful given 
your fAlpha hypothesis from EDA), and is comparatively robust to scale/outliers — a 
deliberately different "type" of classifier from kNN/SVM, which is valuable since 
your project plan even recommends picking models that make different kinds of errors.

- Why not LDA/QDA/Naive Bayes/logistic regression? You could mention you considered 
them, but they all make a distributional assumption (Gaussian, or feature independence) 
that you have no strong evidence holds for this Hillas-parameter data (and your 
EDA showed several features with non-symmetric, skewed distributions) — whereas 
kNN, SVM-RBF, and Random Forest make no such assumption.

---
---
---

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
risk) to larger (k=25, smoother, more underfitting risk), matching the
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

---
---
---

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