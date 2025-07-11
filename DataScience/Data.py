# 🍇 Wine Quality Prediction Project

## 📌 Project Overview
# This notebook explores red and white Vinho Verde wines from Portugal. Our goal is to understand
# which physicochemical properties influence perceived quality and build models to predict wine quality.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, mean_squared_error, classification_report

# Load datasets
red = pd.read_csv(r"C:\Users\gavin\Downloads\winequality-red.csv", sep=';')

white = pd.read_csv(r"C:\Users\gavin\Downloads\winequality-white.csv", sep=';')

# Add a 'type' column to distinguish red/white
red['type'] = 'red'
white['type'] = 'white'

# Combine datasets
wine = pd.concat([red, white], ignore_index=True)

# Basic overview
print(wine.info())
print(wine.describe())

# Plot quality distribution by wine type
sns.countplot(data=wine, x='quality', hue='type')
plt.title("Wine Quality Distribution by Type")
plt.show()

# Create alcohol_cat based on mean ± std
for wine_type in ['red', 'white']:
    avg = wine[wine['type'] == wine_type]['alcohol'].mean()
    std = wine[wine['type'] == wine_type]['alcohol'].std()
    wine.loc[wine['type'] == wine_type, 'alcohol_cat'] = pd.cut(
        wine.loc[wine['type'] == wine_type, 'alcohol'],
        bins=[-float("inf"), avg - std, avg + std, float("inf")],
        labels=['low', 'mid', 'high']
    )

# Plot: Quality vs Alcohol Category
sns.boxplot(data=wine, x='alcohol_cat', y='quality', hue='type')
plt.title("Quality by Alcohol Category and Type")
plt.show()

# Create isSweet variable
threshold = wine['residual sugar'].median()
wine['isSweet'] = (wine['residual sugar'] > threshold).astype(int)

# Plot: Quality by isSweet
sns.countplot(data=wine, x='quality', hue='isSweet')
plt.title("Quality Distribution by Sweetness")
plt.show()

# Correlation matrix
corr = wine.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Feature selection (example)
features = ['alcohol', 'sulphates', 'volatile acidity']  # Update based on correlation matrix

# Binary classification target
wine['quality_binary'] = (wine['quality'] >= 6).astype(int)

# Train/test split
X = wine[features]
y = wine['quality_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Binary classification model
model = RandomForestClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Evaluation
print("F1 Score:", f1_score(y_test, preds))
print("AUC:", roc_auc_score(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# Regression model
y_reg = wine['quality']
regressor = RandomForestRegressor()
regressor.fit(X_train, y_reg.loc[y_train.index])
reg_preds = regressor.predict(X_test)
print("RMSE:", (mean_squared_error(y_reg.loc[y_test.index], reg_preds)) ** 0.5)


# Multi-class classification
model_multi = RandomForestClassifier()
model_multi.fit(X_train, y_reg.loc[y_train.index])
preds_multi = model_multi.predict(X_test)
print(classification_report(y_reg.loc[y_test.index], preds_multi))

# Summary of key findings can be written here as markdown in the final report.
