# =============================================================================
# 01_eda.py — Exploratory Data Analysis (EDA)
# =============================================================================
# WHAT IS THIS FILE?
# This is the first step of any machine learning project: understanding your
# data before touching any algorithm. We look at the structure, distributions,
# class balance, and relationships between features.
#
# WHY DO WE DO THIS FIRST?
# Every decision we make later (which model to use, how to preprocess, which
# metric to evaluate with) must be justified by what we find here. The
# professor grades reasoning, not just results — this file builds that reasoning.
# =============================================================================


# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
# We import the libraries we need. Think of libraries as toolboxes —
# each one gives us ready-made tools so we don't have to write everything
# from scratch.

import pandas as pd          # pandas: for loading and working with tabular data
import numpy as np           # numpy: for numerical operations
import matplotlib.pyplot as plt  # matplotlib: for creating plots
import seaborn as sns        # seaborn: for nicer statistical plots
import os                    # os: for working with file paths and folders

# This line makes all our plots look slightly cleaner by default
sns.set_theme(style="whitegrid")


# =============================================================================
# SECTION 2: LOAD THE DATA
# =============================================================================
# The dataset is a plain text file with comma-separated values but NO header
# row (no column names written in the file itself). We have to supply the
# column names manually, which we know from reading magic04.names.

# Define the column names exactly as described in magic04.names
column_names = [
    "fLength",   # major axis of ellipse [mm]
    "fWidth",    # minor axis of ellipse [mm]
    "fSize",     # 10-log of sum of content of all pixels [#phot]
    "fConc",     # ratio of sum of two highest pixels over fSize
    "fConc1",    # ratio of highest pixel over fSize
    "fAsym",     # distance from highest pixel to center, projected onto major axis [mm]
    "fM3Long",   # 3rd root of third moment along major axis [mm]
    "fM3Trans",  # 3rd root of third moment along minor axis [mm]
    "fAlpha",    # angle of major axis with vector to origin [deg]
    "fDist",     # distance from origin to center of ellipse [mm]
    "class"      # target label: 'g' = gamma (signal), 'h' = hadron (background)
]

# Load the data from the file
# pd.read_csv works for any comma-separated file, not just .csv files
df = pd.read_csv(
    os.path.join("data", "magic04.data"),  # path to the file
    header=None,                            # the file has no header row
    names=column_names                      # so we supply the names ourselves
)

print("=" * 60)
print("DATA LOADED SUCCESSFULLY")
print("=" * 60)


# =============================================================================
# SECTION 3: BASIC STRUCTURE
# =============================================================================
# Before anything else, we check the basic facts about our dataset:
# how many rows, how many columns, what types of data, any missing values?

print("\n--- Shape of the dataset ---")
print(f"Rows (samples):    {df.shape[0]}")
print(f"Columns (features + target): {df.shape[1]}")

# WHAT TO LOOK FOR:
# We expect 19020 rows and 11 columns (10 features + 1 class label).
# If the numbers are wrong, the file didn't load correctly.

print("\n--- Data types of each column ---")
print(df.dtypes)

# WHAT TO LOOK FOR:
# All feature columns should be float64 (decimal numbers).
# The 'class' column should be object (text: 'g' or 'h').
# If something looks wrong here, the column names may be misaligned.

print("\n--- First 5 rows ---")
print(df.head())

# WHAT TO LOOK FOR:
# This just gives us a quick visual sanity check that the data loaded
# correctly and columns are aligned with their names.

print("\n--- Missing values per column ---")
print(df.isnull().sum())

# WHAT TO LOOK FOR:
# We expect 0 missing values in every column.
# The UCI page confirms there are no missing values in this dataset.
# If we see any non-zero numbers here, something went wrong with loading.


# =============================================================================
# SECTION 4: CLASS DISTRIBUTION
# =============================================================================
# Our target variable has two classes: 'g' (gamma, signal) and 'h' (hadron,
# background). We need to check how many samples belong to each class.
#
# WHY DOES THIS MATTER?
# If one class has far more samples than the other (called "class imbalance"),
# then a naive model can cheat — it can predict the majority class every time
# and still get high accuracy. For example, if 90% of samples are class A,
# always predicting A gives 90% accuracy while being completely useless.
# The MAGIC dataset description explicitly warns us about this.

print("\n--- Class distribution ---")
class_counts = df["class"].value_counts()
class_proportions = df["class"].value_counts(normalize=True) * 100

print("Counts:")
print(class_counts)
print("\nProportions (%):")
print(class_proportions.round(2))

# WHAT TO LOOK FOR:
# g (gamma/signal):   ~12332 samples (~64.8%)
# h (hadron/background): ~6688 samples (~35.2%)
# This is a moderately imbalanced dataset. Not extreme, but enough that
# we CANNOT use accuracy as our main evaluation metric.
# We will use ROC-AUC, sensitivity, and specificity instead.
# The dataset description itself tells us this explicitly.

# --- PLOT: Class distribution bar chart ---
fig, ax = plt.subplots(figsize=(6, 4))

class_counts.plot(
    kind="bar",
    ax=ax,
    color=["steelblue", "coral"],
    edgecolor="black"
)

ax.set_title("Class Distribution: Gamma vs Hadron", fontsize=13)
ax.set_xlabel("Class (g = gamma/signal, h = hadron/background)", fontsize=11)
ax.set_ylabel("Number of Samples", fontsize=11)
ax.set_xticklabels(["g (gamma)", "h (hadron)"], rotation=0)

# Add the count and percentage as text on top of each bar
for i, (count, prop) in enumerate(zip(class_counts, class_proportions)):
    ax.text(i, count + 100, f"{count}\n({prop:.1f}%)",
            ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "01_class_distribution.png"), dpi=150)
plt.show()
print("Plot saved: 01_class_distribution.png")


# =============================================================================
# SECTION 5: SUMMARY STATISTICS PER CLASS
# =============================================================================
# We compute basic statistics (mean, std, min, max) for each feature,
# separated by class. This tells us whether the two classes look different
# in terms of their feature values — which is a prerequisite for any
# classifier to work at all.

feature_cols = [col for col in df.columns if col != "class"]
# This creates a list of all column names except the target column.
# Result: ['fLength', 'fWidth', 'fSize', ..., 'fDist']

print("\n--- Summary statistics for GAMMA (g) class ---")
print(df[df["class"] == "g"][feature_cols].describe().round(3))

print("\n--- Summary statistics for HADRON (h) class ---")
print(df[df["class"] == "h"][feature_cols].describe().round(3))

# WHAT TO LOOK FOR:
# Compare the means between the two classes for each feature.
# If a feature has very different means for g vs h, it is likely useful
# for discrimination. If the means are nearly identical, the feature
# may not help much.
# fAlpha in particular is known from physics to be highly discriminative:
# gamma showers point toward the telescope source, hadronic showers don't.


# =============================================================================
# SECTION 6: FEATURE DISTRIBUTIONS PER CLASS
# =============================================================================
# We plot a histogram for each feature, with the two classes overlaid.
# This gives us a visual version of the summary statistics above.
#
# WHY HISTOGRAMS?
# They show the full shape of the distribution, not just the mean.
# Two features can have different means but completely overlapping
# distributions — in which case a classifier will struggle to use them.
# Conversely, even slight separation in the distributions is useful.

fig, axes = plt.subplots(
    nrows=2, ncols=5,       # 2 rows, 5 columns = 10 subplots, one per feature
    figsize=(18, 7)
)
axes = axes.flatten()       # flatten from 2D array to 1D so we can loop over it

for i, col in enumerate(feature_cols):
    ax = axes[i]

    # Plot histogram for gamma class
    ax.hist(
        df[df["class"] == "g"][col],
        bins=50,
        alpha=0.5,          # alpha controls transparency (0=invisible, 1=solid)
        color="steelblue",
        label="gamma (g)",
        density=True        # density=True normalizes so both classes are comparable
                            # even though they have different numbers of samples
    )

    # Plot histogram for hadron class
    ax.hist(
        df[df["class"] == "h"][col],
        bins=50,
        alpha=0.5,
        color="coral",
        label="hadron (h)",
        density=True
    )

    ax.set_title(col, fontsize=11)
    ax.set_xlabel("Value", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)

plt.suptitle(
    "Feature Distributions by Class (normalized)",
    fontsize=14,
    y=1.02
)
plt.tight_layout()
plt.savefig(
    os.path.join("results", "plots", "02_feature_distributions.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.show()
print("Plot saved: 02_feature_distributions.png")

# WHAT TO LOOK FOR:
# - fAlpha: should show very clear separation — gamma peaks near 0 degrees
#   (the shower points toward the source), hadron is more uniform.
# - fSize, fLength, fWidth: moderate separation expected.
# - fAsym, fM3Long, fM3Trans: these are asymmetry measures; separation
#   may be less obvious but still present.
# Features where the two histograms heavily overlap will be harder for
# classifiers to use. Features with clear separation are more valuable.


# =============================================================================
# SECTION 7: CORRELATION MATRIX
# =============================================================================
# A correlation matrix shows how strongly each pair of features is linearly
# related to each other. Values close to 1 or -1 = strong relationship.
# Values close to 0 = little to no linear relationship.
#
# WHY DOES THIS MATTER?
# If two features are highly correlated (e.g. 0.95), they carry almost
# the same information. Including both doesn't help much and can cause
# problems (called multicollinearity). This informs feature selection later.
#
# IMPORTANT WARNING — correlation ≠ causation:
# Just because two features move together doesn't mean one causes the other.
# Your professor gave the shark attack / ice cream example in Lecture 7.
# We use correlation to spot redundancy, not to infer meaning.

corr_matrix = df[feature_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,         # show the correlation value in each cell
    fmt=".2f",          # round to 2 decimal places
    cmap="coolwarm",    # blue = negative correlation, red = positive
    center=0,           # center the colormap at 0
    square=True,        # make each cell square
    linewidths=0.5,
    ax=ax
)
ax.set_title("Feature Correlation Matrix", fontsize=13)
plt.tight_layout()
plt.savefig(
    os.path.join("results", "plots", "03_correlation_matrix.png"),
    dpi=150
)
plt.show()
print("Plot saved: 03_correlation_matrix.png")

# WHAT TO LOOK FOR:
# - fConc and fConc1 are likely highly correlated (both measure pixel
#   concentration ratios). If correlation > 0.9, consider noting this.
# - fLength and fWidth may correlate (both are ellipse axis measurements).
# - fAsym, fM3Long, fM3Trans may have lower correlations with others
#   since they measure different geometric properties.


# =============================================================================
# SECTION 8: SCATTER PLOTS FOR KEY FEATURE PAIRS
# =============================================================================
# We pick a few of the most physically meaningful feature pairs and plot
# them as scatter plots, coloring points by class. This is the clearest
# visual test of whether classes are separable.
#
# We can't plot all 45 possible pairs (10 features = 45 combinations),
# so we select the most informative ones based on the histogram findings.

# We use a sample of the data to keep the plot readable
# (19020 points overlapping = hard to see anything)
sample = df.sample(n=2000, random_state=42)
# random_state=42 means the random sample is reproducible —
# running the code again gives the same sample.

pairs_to_plot = [
    ("fAlpha", "fLength"),
    ("fAlpha", "fSize"),
    ("fWidth", "fLength"),
    ("fConc", "fSize"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (x_col, y_col) in enumerate(pairs_to_plot):
    ax = axes[i]

    for cls, color, label in [("g", "steelblue", "gamma"), ("h", "coral", "hadron")]:
        subset = sample[sample["class"] == cls]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            alpha=0.3,      # low alpha because many points overlap
            s=10,           # small point size
            color=color,
            label=label
        )

    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.set_title(f"{x_col} vs {y_col}", fontsize=12)
    ax.legend(fontsize=9)

plt.suptitle(
    "Scatter Plots of Key Feature Pairs (sample of 2000)",
    fontsize=14
)
plt.tight_layout()
plt.savefig(
    os.path.join("results", "plots", "04_scatter_plots.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.show()
print("Plot saved: 04_scatter_plots.png")

# WHAT TO LOOK FOR:
# - fAlpha vs anything: gamma should cluster near low fAlpha values
#   (near 0 degrees), hadron more spread out. This is the most important
#   discriminating feature physically.
# - If you see clear cluster separation in a pair, that pair will be
#   useful for classifiers that use distance (kNN, SVM).
# - If the two classes completely overlap, those features alone won't help.


# =============================================================================
# SECTION 9: EDA SUMMARY — WRITE YOUR FINDINGS
# =============================================================================
# This is where you document what you actually observed. These observations
# directly justify your decisions in the preprocessing and modelling steps.
# The professor grades this reasoning chain, so be explicit.

print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print("""
DATASET:
- 19,020 samples, 10 continuous features, 1 binary target
- No missing values
- All features are real-valued (float) — no encoding needed for features
- Target: 'g' (gamma/signal) and 'h' (hadron/background)

CLASS IMBALANCE:
- ~64.8% gamma, ~35.2% hadron
- Classes are imbalanced — accuracy is NOT a reliable metric here
- We will use ROC-AUC as the primary evaluation metric, supported by
  sensitivity and specificity at relevant thresholds
- We will use stratified splits to preserve the class ratio in all splits

FEATURE OBSERVATIONS:
- fAlpha shows the clearest class separation, consistent with its
  physical meaning: gamma showers point toward the source (low alpha),
  hadronic showers do not (more uniform distribution)
- fSize, fLength, fWidth show moderate separation
- fConc and fConc1 are likely highly correlated (both measure pixel
  concentration) — worth noting but we keep both for now

PREPROCESSING IMPLICATIONS:
- Feature scaling IS required: our models include kNN (distance-based)
  and SVM (margin-based) — both are sensitive to feature scale
- Random Forest does NOT require scaling, but we apply it anyway for
  consistency across models
- Scaler must be fit ONLY on training data to avoid data leakage
""")