# 🍇 Wine Quality Prediction Project

## 📌 Project Overview
This project analyzes red and white Vinho Verde wines from Portugal to understand which physicochemical properties influence perceived quality and builds machine learning models to predict wine quality. It includes exploratory data analysis, feature engineering, and models for binary classification, regression, and multi-class classification.

## 🚀 Features
- Visualizations of quality distributions by wine type, alcohol categories, and sweetness.
- Correlation analysis to select important features.
- Binary classification model predicting if a wine is good quality (quality ≥ 6).
- Regression model predicting exact quality scores.
- Multi-class classification for detailed quality prediction.
- Evaluation using F1 Score, AUC, RMSE, and classification reports.

## 🛠 Installation
1. Clone the repo:
git clone https://github.com/GavinNjoyo/Wine-Machine-Learning.git
cd Wine-Machine-Learning


2. (Optional) Create and activate a virtual environment:
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate


3. Install dependencies:
pip install -r requirements.txt

## 📚 Usage
1. Place `winequality-red.csv` and `winequality-white.csv` in the project folder or update file paths in the code.
2. Run the notebook or script:
jupyter notebook Wine_Quality_Analysis.ipynb





## 📈 Results Summary
- Binary classifier F1 Score: *[insert your result]*
- Binary classifier AUC: *[insert your result]*
- Regression RMSE: *[insert your result]*
- Multi-class classification report available in the notebook.

## 📊 Visualizations
- Quality distribution by wine type
- Quality vs alcohol category and sweetness
- Correlation heatmap of features

## 🧰 Technologies
Python, Pandas, Scikit-learn, Seaborn, Matplotlib, Jupyter Notebook

## 📂 Data Source
Datasets from UCI Machine Learning Repository: [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)

## 🤝 Contributing
Feel free to open issues or submit pull requests!

## 🔗 Link to Code
[Click here to view the GitHub repository](https://github.com/GavinNjoyo/Wine-Machine-Learning)
