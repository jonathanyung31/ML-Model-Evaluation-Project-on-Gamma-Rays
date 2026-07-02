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