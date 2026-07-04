import pandas as pd          # pandas: for loading and working with tabular data
import numpy as np           # numpy: for numerical operations
import matplotlib.pyplot as plt  # matplotlib: for creating plots
import seaborn as sns        # seaborn: for nicer statistical plots
import os                    # os: for working with file paths and folders

# This line makes all our plots look slightly cleaner by default
sns.set_theme(style="darkgrid")
plt.style.use("dark_background")
plt.rcParams["grid.color"] = "#3a3a3a"   # dark grey grid lines
plt.rcParams["grid.alpha"] = 0.6         # optional: slightly transparent

column_names = [
    "fLength",   
    "fWidth",    
    "fSize",     
    "fConc",     
    "fConc1",    
    "fAsym",     
    "fM3Long",   
    "fM3Trans",  
    "fAlpha",    
    "fDist",     
    "class"      
]


## SECTION 2: LOAD THE DATA

df = pd.read_csv(
    os.path.join("data", "magic04.data"),  # path to the file
    header=None,                            # the file has no header row
    names=column_names                      # so we supply the names ourselves
)

print("DATA LOADED SUCCESSFULLY")


df = df.copy()

# Check for and remove exact duplicate rows before any exploration.
# See reasoning: with CORSIKA-simulated continuous features, an exact
# match across all 10 decimals is a data-pipeline artifact, not a real
# repeated physical event (full justification kept in Section 3 notes).
n_duplicates = df.duplicated().sum()
print(f"Duplicate rows found: {n_duplicates}")


# SECTION 3: BASIC STRUCTURE

print("\n--- Shape of the dataset ---")
print(f"Rows (samples): {df.shape[0]}")     # number of rows
print(f"Columns (features + target): {df.shape[1]}")    # number of columns


print("\n--- Data types of each column ---")
print(df.dtypes)

print("\n--- First 5 rows ---")
print(df.head())


print("\n--- Missing values per column ---")
print(df.isnull().sum())


print("\n--- Basic statistics ---")
print(df.describe())


print("\n--- Unique values in target column ---")
print(df["class"].unique())


print("\n--- Duplicate rows ---")
print(f"Number of duplicate rows: {df.duplicated().sum()}")


print("\n--- Checking for hidden missing-like values ---")
print((df == "?").sum())
print((df == "NA").sum())
print((df == "null").sum())
print((df == "").sum())

# --- Slide-ready summary table: only the features you'll actually discuss ---
key_features = ["fAlpha", "fSize", "fLength", "fWidth"]
summary_table = pd.concat([
    df[df["class"] == "g"][key_features].mean().rename("gamma_mean"),
    df[df["class"] == "h"][key_features].mean().rename("hadron_mean"),
], axis=1).round(2)

print(summary_table)
# Recreate this as a clean table in your slide software — 4 rows, 2 columns,
# 2 decimal places, large font. Do not screenshot the console.


# SECTION 4: CLASS DISTRIBUTION

print("\n--- Class distribution ---")
class_counts = df["class"].value_counts()
class_proportions = df["class"].value_counts(normalize=True) * 100     # in precentages

print("Counts:")
print(class_counts)
print("\nProportions (%):")
print(class_proportions.round(2))   # limiting to 2 decimal points


# PLOT: Class distribution bar chart
fig, ax = plt.subplots(figsize=(6, 4))

class_counts.plot(
    kind="bar",
    ax=ax,
    color=["deepskyblue", "magenta"],
    edgecolor="black"
)

ax.set_ylim(0, max(class_counts) * 1.25)    # cahnge y-axis limits for avoiding overlapping text

ax.set_title("Class Distribution: Gamma vs Hadron", fontsize=13)

# Replace the default x-axis labels with more descriptive labels
# rotation=0 keeps the labels horizontal
ax.set_xticklabels(["g (gamma)", "h (hadron)"], rotation=0)

ax.set_xlabel("Class (g = gamma/signal, h = hadron/background)", fontsize=12, labelpad=12)
ax.set_ylabel("Number of Samples", fontsize=12, labelpad=12)

# Add the count and percentage as text on top of each bar
for i, (count, prop) in enumerate(zip(class_counts, class_proportions)):
    # Place text at x-position i and slightly above the bar height.
    # \n creates a new line so the count and percentage appear on
    # separate lines.
    ax.text(i, count + 100, f"{count}\n({prop:.1f}%)",
            ha="center",        # horizontally center the text
            va="bottom",        # align the bottom of the text to the position
            fontsize=10)

plt.tight_layout()      # Automatically adjust spacing so labels and titles don't overlap
# Save the figure to a PNG file with good image quality
# dpi=150 means 150 dots (pixels) per inch
plt.savefig(os.path.join("results", "plots", "01_class_distribution.png"), dpi=150)
plt.show()      # Display the plot on the screen
print("Plot saved: 01_class_distribution.png")

# SECTION 5: SUMMARY STATISTICS PER CLASS

feature_cols = [col for col in df.columns if col != "class"]
# This creates a list of all column names except the target column.

# describe() but only on rows with "g"
print("\n--- Summary statistics for GAMMA (g) class ---")
print(df[df["class"] == "g"][feature_cols].describe().round(3))

# describe() but only on rows with "h"
print("\n--- Summary statistics for HADRON (h) class ---")
print(df[df["class"] == "h"][feature_cols].describe().round(3))

# SECTION 6: FEATURE DISTRIBUTIONS PER CLASS

fig, axes = plt.subplots(
    nrows=2, ncols=5,       # 2 rows, 5 columns = 10 subplots, one per feature
    figsize=(18, 7)
)
axes = axes.flatten()       # flatten from 2D array to 1D so we can loop over it

for i, col in enumerate(feature_cols):      # go through everything that is not the target
    ax = axes[i]


    # Plot histogram for gamma class
    ax.hist(
        df[df["class"] == "g"][col],
        bins=50,
        alpha=0.85,          # alpha controls transparency (0=invisible, 1=solid)
        color="deepskyblue",
        label="gamma (g)",
        density=True        # density=True normalizes so both classes are comparable
                            # even though they have different numbers of samples
    )

    # Plot histogram for hadron class
    ax.hist(
        df[df["class"] == "h"][col],
        bins=50,
        alpha=0.6,
        color="magenta",
        label="hadron (h)",
        density=True
    )

    ax.set_title(col, fontsize=11)
    ax.set_xlabel("Value", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)

plt.suptitle(       # super title for the entire figure
    "Feature Distributions by Class (normalized)",
    fontsize=14,
    y=0.98  # where it's placed here
)

plt.tight_layout()
plt.savefig(
    os.path.join("results", "plots", "02_feature_distributions.png"),
    dpi=150,        # saved in better resolution
    bbox_inches="tight"     
)
plt.show()
print("Plot saved: 02_feature_distributions.png")

# --- Slide-ready version: only the 2 most informative features ---
# Chosen from the full grid: fAlpha (strongest separator) and fSize
# (weaker than expected -> good contrast point for discussion)
slide_features = ["fAlpha", "fSize"]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4.5))

for ax, col in zip(axes, slide_features):
    ax.hist(df[df["class"] == "g"][col], bins=40, alpha=0.85,
             color="deepskyblue", label="gamma (g)", density=True)
    ax.hist(df[df["class"] == "h"][col], bins=40, alpha=0.6,
             color="blueviolet", label="hadron (h)", density=True)
    ax.set_title(col, fontsize=13)
    ax.set_xlabel("Value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)

plt.suptitle("Feature Separation: Strongest (fAlpha) vs Weaker (fSize)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join("results", "plots", "02b_slide_distributions.png"), dpi=150, bbox_inches="tight")
plt.show()


# SECTION 7: CORRELATION MATRIX


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



# SECTION 8: SCATTER PLOTS FOR KEY FEATURE PAIRS

pairs_to_plot = [
    ("fAlpha", "fLength"),
    ("fAlpha", "fWidth"),
    ("fAlpha", "fSize"),
    ("fWidth", "fLength"),
    ("fConc", "fSize"),
    ("fConc", "fConc1"),
]

# Plotting full data is too much to see difference
sample = df.sample(n=2000, random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, (x_col, y_col) in enumerate(pairs_to_plot):
    ax = axes[i]

    for cls, color, label in [("g", "deepskyblue", "gamma"), ("h", "magenta", "hadron")]:
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
