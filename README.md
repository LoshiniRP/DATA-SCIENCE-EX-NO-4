# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file.

# ALGORITHM:
STEP 1: Read the given Data.

STEP 2: Clean the Data Set using Data Cleaning Process.

STEP 3: Apply Feature Scaling for the feature in the data set.

STEP 4: Apply Feature Selection for the feature in the data set.

STEP 5: Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

The feature selection techniques used are:

1. Filter Method

2. Wrapper Method

3. Embedded Method

# CODING AND OUTPUT:
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
df = pd.read_csv('D:\Bmi.csv') 
print(df.head())

# Handle Missing Values
df = df.dropna()

# Standard Scaling
df_std = df.copy()
scaler_std = StandardScaler()
df_std[['Height', 'Weight']] = scaler_std.fit_transform(df_std[['Height', 'Weight']])
print(df_std.head())

# Min-Max Scaling
df_minmax = df.copy()
scaler_minmax = MinMaxScaler()
df_minmax[['Height', 'Weight']] = scaler_minmax.fit_transform(df_minmax[['Height', 'Weight']])
print(df_minmax.head())

# MaxAbs Scaling
df_maxabs = df.copy()
scaler_maxabs = MaxAbsScaler()
df_maxabs[['Height', 'Weight']] = scaler_maxabs.fit_transform(df_maxabs[['Height', 'Weight']])
print(df_maxabs.head())

# Robust Scaling
df_robust = df.copy()
scaler_robust = RobustScaler()
df_robust[['Height', 'Weight']] = scaler_robust.fit_transform(df_robust[['Height', 'Weight']])
print(df_robust.head())

# Save scaled datasets
df_std.to_csv("BMI_StandardScaled.csv", index=False)
df_minmax.to_csv("BMI_MinMaxScaled.csv", index=False)
df_maxabs.to_csv("BMI_MaxAbsScaled.csv", index=False)
df_robust.to_csv("BMI_RobustScaled.csv", index=False)

# Convert categorical column (Gender) to numeric
df_fs = df.copy()
df_fs['Gender'] = df_fs['Gender'].map({'Male': 0, 'Female': 1})

# Features and target
X = df_fs[['Gender', 'Height', 'Weight']]
y = df_fs['Index']   # target column

from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=2)
X_filter = selector.fit_transform(X, y)

print(X.columns[selector.get_support()])

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=5000, solver='liblinear')

rfe = RFE(model, n_features_to_select=2)
X_wrapper = rfe.fit_transform(X, y)

print(X.columns[rfe.get_support()])

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.01)
lasso.fit(X, y)

selected_features = X.columns[lasso.coef_ != 0]

print(selected_features)

# Save dataset with selected features
selected_cols = X.columns[selector.get_support()]
df_selected = df_fs[selected_cols.tolist() + ['Index']]

df_selected.to_csv("BMI_SelectedFeatures.csv", index=False)
```

<img width="1341" height="600" alt="image" src="https://github.com/user-attachments/assets/67937d3d-227f-4599-989c-75f886d7f4c6" />

<img width="1342" height="558" alt="image" src="https://github.com/user-attachments/assets/e04109ed-8a3e-4b67-9a92-b4645576ceda" />

<img width="1339" height="721" alt="image" src="https://github.com/user-attachments/assets/ec7d6b64-0b83-403d-91f4-dc069d595f6a" />


# RESULT:
Thus, the given data and perform Feature Scaling and Feature Selection process and save the data to a file has been completed successfully.
