"""
===========================================================
File        : predict.py
Project     : AI Powered Fraud Detection Platform
Author      : Antony Selvamuthu

Description
-----------
This module loads the trained Machine Learning model and
predicts whether a transaction is fraudulent.

Functions
---------
1. load_model()
2. validate_input()
3. calculate_risk()
4. predict_transaction()

This file is used by

✔ Streamlit Dashboard
✔ FastAPI
✔ LangChain Agent
✔ Fraud Analytics

===========================================================
"""

import os
import joblib
import pandas as pd


# ===========================================================
# Model Location
# ===========================================================

MODEL_PATH = "models/fraud_model.pkl"


# ===========================================================
# Load Model
# ===========================================================

def load_model():
    """
    Loads the trained Machine Learning model.
    """

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "\nModel not found.\n"
            "Run train_model.py first."
        )

    model = joblib.load(MODEL_PATH)

    return model


# Load model only once
model = load_model()


# ===========================================================
# Validate User Input
# ===========================================================

def validate_input(transaction):

    required_columns = [

        "Amount",

        "Transaction_Hour",

        "Transaction_Type",

        "Merchant_Category",

        "Device_Type",

        "Customer_Age",

        "Previous_Fraud"

    ]

    for column in required_columns:

        if column not in transaction:

            raise ValueError(f"Missing Input : {column}")


# ===========================================================
# Risk Level
# ===========================================================

def calculate_risk(probability):

    score = round(probability * 100, 2)

    if score >= 90:

        level = "CRITICAL"

        color = "Red"

    elif score >= 75:

        level = "HIGH"

        color = "Orange"

    elif score >= 50:

        level = "MEDIUM"

        color = "Yellow"

    else:

        level = "LOW"

        color = "Green"

    return score, level, color


# ===========================================================
# Fraud Explanation
# ===========================================================

def generate_reason(transaction):

    reasons = []

    if transaction["Amount"] > 75000:

        reasons.append("Large transaction amount.")

    if transaction["Transaction_Hour"] <= 4:

        reasons.append("Transaction during unusual hours.")

    if transaction["Device_Type"] == 0:

        reasons.append("Unknown device detected.")

    if transaction["Previous_Fraud"] == 1:

        reasons.append("Customer has previous fraud history.")

    if transaction["Merchant_Category"] == 5:

        reasons.append("International merchant transaction.")

    if len(reasons) == 0:

        reasons.append("No abnormal activity detected.")

    return reasons


# ===========================================================
# Recommendation
# ===========================================================

def recommendation(score):

    if score >= 90:

        return "Immediately block the transaction and notify the customer."

    elif score >= 75:

        return "Hold the transaction and verify using OTP."

    elif score >= 50:

        return "Monitor the transaction carefully."

    else:

        return "Transaction appears safe."


# ===========================================================
# Prediction Function
# ===========================================================

def predict_transaction(transaction):

    validate_input(transaction)

    dataframe = pd.DataFrame([transaction])

    prediction = model.predict(dataframe)[0]

    probability = model.predict_proba(dataframe)[0][1]

    risk_score, risk_level, color = calculate_risk(probability)

    reasons = generate_reason(transaction)

    advice = recommendation(risk_score)

    result = {

        "Prediction":

            "Fraudulent"
            if prediction == 1
            else "Legitimate",

        "Fraud Probability":

            round(probability, 4),

        "Risk Score":

            risk_score,

        "Risk Level":

            risk_level,

        "Risk Color":

            color,

        "Reasons":

            reasons,

        "Recommendation":

            advice

    }

    return result


# ===========================================================
# Test Program
# ===========================================================

if __name__ == "__main__":

    sample_transaction = {

        "Amount":95000,

        "Transaction_Hour":2,

        "Transaction_Type":1,

        "Merchant_Category":5,

        "Device_Type":0,

        "Customer_Age":30,

        "Previous_Fraud":1

    }

    result = predict_transaction(sample_transaction)

    print("\n")

    print("=" * 60)

    print("AI FRAUD DETECTION RESULT")

    print("=" * 60)

    for key, value in result.items():

        print(f"{key:20} : {value}")

    print("=" * 60)