import streamlit as st
import pandas as pd
from utils.model_loader import load_classification_model, load_regression_model
from utils.preprocessing import preprocess_input
from utils.logger import log_prediction
import joblib
import numpy as np
st.title("🏦 EMI Eligibility & Maximum EMI Predictor")

# Load Models
clf_model = load_classification_model()
reg_model = load_regression_model()

# ------------------------------
# 1️⃣ Personal Demographics
# ------------------------------
st.subheader("1️⃣ Personal Demographics")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 25, 60)
    gender = st.selectbox("Gender", ["MALE", "FEMALE"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])

with col2:
    education = st.selectbox("Education",
                             ["High School", "Graduate",
                              "Post Graduate", "Professional"])

# ------------------------------
# 2️⃣ Employment & Income
# ------------------------------
st.subheader("2️⃣ Employment & Income")
col3, col4 = st.columns(2)

with col3:
    monthly_salary = st.number_input("Monthly Salary (INR)", 15000, 200000)
    employment_type = st.selectbox("Employment Type",
                                    ["Private", "Government", "Self-employed"])
    years_of_employment = st.number_input("Years of Employment")

with col4:
    company_type = st.selectbox("Company Type",
                                 ["Startup","Small", "Mid-size", "Large Indian","MNC"])

# ------------------------------
# 3️⃣ Housing & Family
# ------------------------------
st.subheader("3️⃣ Housing & Family")
col5, col6 = st.columns(2)

with col5:
    house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
    monthly_rent = st.number_input("Monthly Rent")
    family_size = st.number_input("Family Size")

with col6:
    dependents = st.number_input("Dependents")

# ------------------------------
# 4️⃣ Monthly Financial Obligations
# ------------------------------
st.subheader("4️⃣ Monthly Financial Obligations")
school_fees = st.number_input("School Fees")
college_fees = st.number_input("College Fees")
travel_expenses = st.number_input("Travel Expenses")
groceries_utilities = st.number_input("Groceries & Utilities")
other_monthly_expenses = st.number_input("Other Monthly Expenses")

# ------------------------------
# 5️⃣ Financial Status
# ------------------------------
st.subheader("5️⃣ Financial Status & Credit History")
existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
current_emi_amount = st.number_input("Current EMI Amount")
credit_score = st.slider("Credit Score", 300, 850)
bank_balance = st.number_input("Bank Balance")
emergency_fund = st.number_input("Emergency Fund")

# ------------------------------
# 6️⃣ Loan Application Details
# ------------------------------
st.subheader("6️⃣ Loan Application Details")
emi_scenario = st.selectbox("EMI Scenario",['Personal Loan EMI','E-commerce Shopping EMI','Education EMI','Vehicle EMI','Home Appliances EMI'])
requested_amount = st.number_input("Requested Loan Amount")
requested_tenure = st.slider("Requested Tenure (Months)", 6, 120)

# ==============================
# 🚀 PREDICTION BUTTON
# ==============================

if st.button("🔍 Analyze EMI Eligibility"):

    input_data = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure
    }

    try:
        processed_data = preprocess_input(input_data)
        reg_training_columns = joblib.load("D:\AI & ML\EMIPredictionApp\models\\reg_training_columns.pkl")
        reg_processed_data  = processed_data[reg_training_columns]
        clf_training_columns = joblib.load("D:\AI & ML\EMIPredictionApp\models\clf_training_columns.pkl") 
        clf_processed_data = processed_data[clf_training_columns]
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # --- Regression ---
    max_emi = reg_model.predict(reg_processed_data)[0]
    max_emi  = np.expm1(max_emi) #applying log transform
    
    
    # --- Classification ---
    #clf_processed_data.drop(['gender_MALE','marital_status_Single','employment_type_Government','monthly_rent','school_fees','college_fees','travel_expenses','groceries_utilities','other_monthly_expenses','existing_loans'],axis=1,inplace=True)
    class_pred = clf_model.predict(clf_processed_data)[0]
    class_probs = clf_model.predict_proba(clf_processed_data)[0]

    label_map = {
        0: "Eligible",
        1: "High_Risk",
        2: "Not_Eligible"
    }

    eligibility = label_map[class_pred]
    confidence = max(class_probs)

    

    # ==============================
    # 📊 OUTPUT SECTION
    # ==============================

    st.divider()
    st.subheader("📊 Prediction Results")

    colA, colB = st.columns(2)

    with colA:
        st.metric("EMI Eligibility", eligibility)
        st.write(f"Confidence: {round(confidence*100,2)}%")

    with colB:
        st.metric("Maximum Safe Monthly EMI (INR)", round(max_emi,2))

    # Log Prediction
    log_prediction({
        "emi_eligibility": eligibility,
        "max_monthly_emi": max_emi
    })
    