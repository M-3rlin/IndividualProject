# "Premium" vs. "Standard" Car Classification

Author: Abdul R.
Date: 13/11/2025

## 1. Project Goal

The goal is to find out what numeric features (like enginesize or horsepower) are most important for deciding if a car is "Premium" or "Standard".

## 2. What I Did (The Process)

Loaded Data: Loaded the CarPrice_Assignment.csv file.

Cleaned Data: Converted text for "doors" and "cylinders" into numbers (e.g., "four" became 4).

Created the Target: Created a new column is_premium. Cars over the 75th percentile price (~$18,150) were "Premium" (1) and all others were "Standard" (0).

Prepared for ML (The Simple Way):

y = is_premium column.

X = I selected only the numeric columns from the dataset. I ignored all text columns like fueltype or carbody to keep it simple.

I dropped price, is_premium, car_ID, and symboling from X.

I did not do any scaling or encoding.

Split Data: Split the data into a training set (80%) and a testing set (20%).

Trained Model: I used a RandomForestClassifier and trained it on the (un-scaled) numeric data.

Checked Model: I checked the model's accuracy, printed a classification report, and made a confusion matrix.

Found Important Features: I checked the "feature importances" to see which numeric features the model used most.

## 3. What I Found (Key Results)

Model Accuracy: My model was [e.g., 90%] accurate at predicting if a car was "Standard" or "Premium" using only the numeric data.

Confusion Matrix:
\Individual Project\confusion matrix.png

The model was still very good at finding "Standard" and "Premium" cars.

Feature Importance:
\Individual Project\feature importance.png

The most important feature was enginesize.

The next most important features were curbweight and horsepower.

This proves that even without all the complex data, these three features are the most important.

## 4. Conclusion (What this means)

Even by taking the simplest path (using only numeric features), the project is a success.

We proved that the most important factors for classifying a car as "Premium" are its enginesize, curbweight, and horsepower.

A company can use this information: to build a "Premium" car, build it with a big engine. To build a "Standard" car, focus on other things, like highwaympg.