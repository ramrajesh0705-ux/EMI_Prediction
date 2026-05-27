import streamlit as st

st.title("🏦 EMI Risk & Approval Prediction")

st.markdown("""
## Project Overview
This application helps lenders and borrowers understand EMI behavior and loan approval likelihood using data-driven machine learning.
It combines regression and classification models to predict EMI amounts and assess loan approval risk, while also providing interactive visualizations and monitoring tools.

## What this app includes
- **EMI Prediction:** Predicts the monthly EMI amount for a loan based on borrower and loan features.
- **Loan Approval Prediction:** Estimates whether an application is likely to be approved.
- **Data Explorer:** Visualizes distributions, relationships, and trends in the EMI dataset.
- **Model Monitoring:** Tracks model performance and data trends over time.
- **Admin Panel:** Manage logs, model artifacts, and configuration settings.

## Why it matters
- Helps financial teams make faster, more accurate lending decisions.
- Improves risk assessment with automated scoring.
- Provides transparency through charts, metrics, and clear model insights.

## Under the hood
- **Data:** Uses a cleaned EMI dataset with borrower, loan, and credit-related features.
- **Models:** Includes regression and classification pipelines for prediction.
- **Deployment:** Built with Streamlit for a responsive web interface.

## How to use
1. Navigate to the Prediction page to enter loan details and get EMI / approval forecasts.
2. Visit Data Explorer to inspect input features and target distributions.
3. Use Model Monitoring to review model performance and detect drift.
4. Open the Admin Panel for management tools and audit logs.
""")