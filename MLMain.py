import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score


columns = [
    'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class'
]


df = pd.read_csv('data/magic04.data', header=None, names=columns)


# show preview of dataset (requirement: pip install tabulate) only for the wiki
# print(df.head().to_markdown())

print(f"Number of rows and columns: {df.shape}") # 19020 rows and 11 columns


# missing values check

missval = df.isnull().sum()
print(f"Missing values: {missval}") # no  missing values detected

# duplicates?
dupl = df.duplicated().sum()
print(f"Duplicates: {dupl}") # shows 115 duplicates

# check what these duplicate numbers are about
insp = df[df.duplicated()].head(10)
print(insp)

df = df.drop_duplicates()
print(f"Duplicates now: {df.duplicated().sum()}") # 0

# since the duplicates are 115 and only 0.6% of the data it is easier to just drop these



# Class balance, how many examples exist of each class

print(df['class'].value_counts())
print(df['class'].value_counts(normalize=True)) # %

'''
g (gamma) has 12.332
h (background) has 6.573

dtype = float

in % that means g = 65% and h = ~ 34%

'''


# Check for outliers

df.drop(columns=['class']). boxplot(figsize=(14,5), rot=45)
plt.title('Boxplot of all features')
plt.tight_layout()
plt.show()



# define X and y
