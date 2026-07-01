# ================================================================
# File Name : train_model.py
# Project   : AI Powered Fraud Detection Platform
# Author    : Antony Selvamuthu
#
# Description:
# This program trains a Machine Learning model to detect fraudulent
# transactions. It reads a CSV dataset, trains a Random Forest model,
# evaluates its performance, and saves the trained model for future
# predictions.
# ================================================================


# -------------------------------
# Import Required Libraries
# -------------------------------

import os
import joblib
import pandas as pd

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ================================================================
# STEP 1 : Create "models" folder
# ================================================================
# If the folder does not exist, create it automatically.
# The trained model (.pkl file) will be stored here.
# ================================================================

os.makedirs("models", exist_ok=True)


# ================================================================
# STEP 2 : Load Dataset
# ================================================================
# Dataset should be stored inside:
#
# data/transactions.csv
#
# ================================================================

DATASET_PATH = "/content/drive/MyDrive/transactions.csv"

# Check whether dataset exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        "Dataset not found!\n"
        "Please place transactions.csv inside the data folder."
    )

# Read CSV using pandas
df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

# Display first five rows
print(df.head())


# ================================================================
# STEP 3 : Dataset Information
# ================================================================
# Display dataset statistics
# ================================================================

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nData Types")
print(df.dtypes)


# ================================================================
# STEP 4 : Missing Value Check
# ================================================================
# Missing values can reduce model accuracy.
# Here we simply remove rows with missing values.
# ================================================================

print("\nChecking Missing Values...\n")

print(df.isnull().sum())

# Remove missing values
df = df.dropna()

print("\nRows after removing missing values :", len(df))


# ================================================================
# STEP 5 : Select Features
# ================================================================
# Features are the input variables used for prediction.
#
# Fraud is the target/output variable.
# ================================================================

FEATURES = [

    "Amount",

    "Transaction_Hour",

    "Transaction_Type",

    "Merchant_Category",

    "Device_Type",

    "Customer_Age",

    "Previous_Fraud"

]

TARGET = "Fraud"

# X contains inputs
X = df[FEATURES]

# y contains output
y = df[TARGET]


# ================================================================
# STEP 6 : Split Dataset
# ================================================================
# Split dataset into:
#
# 80% Training
# 20% Testing
#
# random_state=42 ensures same result every run.
# stratify keeps fraud/non-fraud distribution balanced.
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Records :", len(X_train))
print("Testing Records  :", len(X_test))


# ================================================================
# STEP 7 : Create Machine Learning Model
# ================================================================
#
# Random Forest
#
# Advantages:
#
# • High Accuracy
# • Handles large datasets
# • Less overfitting
# • Works well for fraud detection
#
# ================================================================

model = RandomForestClassifier(

    n_estimators=200,

    max_depth=10,

    random_state=42

)


# ================================================================
# STEP 8 : Train Model
# ================================================================
# Model learns fraud patterns from historical transactions.
# ================================================================

print("\nTraining Random Forest Model...\n")

model.fit(X_train, y_train)

print("Training Completed Successfully")


# ================================================================
# STEP 9 : Test Model
# ================================================================
# Predict fraud for testing data.
# ================================================================

y_prediction = model.predict(X_test)


# ================================================================
# STEP 10 : Evaluate Model
# ================================================================
# Calculate:
#
# Accuracy
# Classification Report
# Confusion Matrix
#
# ================================================================

accuracy = accuracy_score(

    y_test,

    y_prediction

)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy : {accuracy*100:.2f}%")

print("\nClassification Report")

print(

    classification_report(

        y_test,

        y_prediction

    )

)

print("\nConfusion Matrix")

print(

    confusion_matrix(

        y_test,

        y_prediction

    )

)


# ================================================================
# STEP 11 : Save Trained Model
# ================================================================
#
# Save model as:
#
# models/fraud_model.pkl
#
# This file will later be used by predict.py
#
# ================================================================

MODEL_PATH = "models/fraud_model.pkl"

joblib.dump(

    model,

    MODEL_PATH

)

print("\nModel Saved Successfully")

print("Location :", MODEL_PATH)


# ================================================================
# STEP 12 : Feature Importance
# ================================================================
# Random Forest tells us which features contribute the most.
# ================================================================

print("\nFeature Importance")

importance = model.feature_importances_

for feature, score in zip(FEATURES, importance):

    print(f"{feature:25} : {score:.4f}")


# ================================================================
# END OF PROGRAM
# ================================================================

print("\n" + "=" * 60)
print("AI FRAUD DETECTION MODEL TRAINING COMPLETED")
print("=" * 60)