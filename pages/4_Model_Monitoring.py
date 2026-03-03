import streamlit as st
import pandas as pd

st.title("📡 Model Monitoring Dashboard")

df = pd.read_csv(
    "data/prediction_logs.csv",
    header=None,
    names=["prediction", "max_emi", "timestamp"]
)

st.metric("Total Predictions", len(df))

if "prediction" in df.columns:
    st.metric("Unique Prediction Types", df["prediction"].nunique())

st.line_chart(df["timestamp"].value_counts())