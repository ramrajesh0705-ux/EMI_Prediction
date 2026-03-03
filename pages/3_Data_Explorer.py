import streamlit as st
import pandas as pd
from utils.charts import prediction_distribution_chart

st.title("📊 Data Explorer")

df = pd.read_csv(
    "data/prediction_logs.csv",
    header=None,
    names=["prediction", "max_emi", "timestamp"]
)

st.dataframe(df)

fig = prediction_distribution_chart(df)
st.plotly_chart(fig)