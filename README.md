# "Premium" vs. "Standard" Car Price Classification

This is a data analytics project for the Code Institute 16-week Data Analytics and AI Bootcamp. The project analyses a dataset of US car prices to build a machine learning model that can classify a vehicle as "Premium" or "Standard".

This project is based on the "ultra-simple" methodology, which uses only numeric features to build a powerful and interpretable model.

## 1. Project Goal

The primary goal is to identify the key technical specifications that differentiate a "Premium" car from a "Standard" car in the US market. We aim to answer the business question: "What features are the most important for classifying a car as premium?"

## 2. Dataset

The project uses the Car Price Prediction dataset, which is publicly available on Kaggle:

Data Source: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction

File: CarPrice_Assignment.csv

## 3. Methodology

The analysis was performed using Python with the pandas, scikit-learn, and seaborn libraries. The process was simplified to focus on the most impactful features:

Load Data: The CarPrice_Assignment.csv file is loaded into a pandas DataFrame.

Clean Data: Text-based numbers (e.g., "four", "six" for doors and cylinders) are converted to numeric integers (4, 6).

Feature Engineering (Target): A new target column, is_premium, is created.

The 75th percentile price is calculated (approx. $18,150).

Cars with a price above this threshold are labeled "Premium" (1).

All other cars are labeled "Standard" (0).

Prepare for ML (Ultra-Simple):

y (target) is set to the is_premium column.

X (features) is created by selecting only the numeric columns from the dataset.

All text-based categorical columns (like fueltype, carbody) are ignored to simplify the model.

No feature scaling (like StandardScaler) is used, as the RandomForestClassifier is robust to un-scaled data.

Train Model: The data is split (80% train, 20% test), and a RandomForestClassifier is trained on the numeric features.

Evaluate: The model's performance is evaluated using an accuracy score, classification report, and a confusion matrix.

Find Key Drivers: The model's feature_importances_ are extracted to identify which features were most predictive.

## 4. Key Findings

Model Performance: The model achieves an accuracy of ~90-93% using only numeric features.

Key "Premium" Drivers: The most important features for classifying a car as "Premium" are, in order:

enginesize

curbweight

horsepower

Key "Standard" Drivers: High highwaympg and citympg are strong indicators of a "Standard" vehicle.

This proves that the market is clearly segmented based on core performance and size metrics.

## 5. How to Run This Project

This project is best run in a Jupyter Notebook environment (like one launched from Anaconda Navigator).

Prerequisites

Anaconda (or Python 3.x)

Jupyter Notebook

Libraries

You must have the following libraries installed. If you are using Anaconda, they should all be included in the (base) environment.

pandas

numpy

matplotlib

seaborn

scikit-learn

If you are not using Anaconda, you can install them via pip:

pip install pandas numpy matplotlib seaborn scikit-learn


## 6. Project Files

classification_project.ipynb: The main Python script with all the analysis.

classififcation_project.py: The main Python script without analysis - raw code and comments

report.md: The final project report explaining the findings.

CarPrice_Assignment.csv: The raw dataset used for the analysis.

README.md: This file.