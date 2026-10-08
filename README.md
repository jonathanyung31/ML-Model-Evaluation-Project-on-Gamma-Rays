# Machine Learning 26
Team: Jonathan Yung, Vanessa Hozanovic
Matr. Nr.: ...

# Project Description



# Requirements
We import the libraries we need:

- pandas
- numpy
- matplotlib
- seaborn
- os

# pandas: for loading and working with tabular data
# numpy: for numerical operations
# matplotlib: for creating plots
# seaborn: for nicer statistical plots
# os: for working with file paths and folders


# Dataset
For this Project, we are using the dataset that can be found here: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

# EDA SUMMARY

DATASET:
- 19,020 samples, 10 continuous features, 1 binary target
- No missing values
- All features are real-valued (float) — no encoding needed for features
- Target: 'g' (gamma/signal) and 'h' (hadron/background)

CLASS IMBALANCE:
- ~64.8% gamma, ~35.2% hadron
- Classes are imbalanced - accuracy is NOT a reliable metric here
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




