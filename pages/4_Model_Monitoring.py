import os
from datetime import timedelta

import pandas as pd
import streamlit as st

#from config.settings import PREDICTION_LOG_PATH
PREDICTION_LOG_PATH = "data/prediction_logs.csv"
st.title("📡 Model Monitoring Dashboard")
st.markdown(
    "This dashboard shows prediction log volume, outcome distribution, and recent model behavior over time."
)


@st.cache_data
def load_prediction_logs(log_path: str) -> pd.DataFrame:
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=["prediction", "max_emi", "timestamp"])

    df = pd.read_csv(
        log_path,
        header=None,
        names=["prediction", "max_emi", "timestamp"],
        parse_dates=[2],
        infer_datetime_format=True,
        on_bad_lines="skip",
    )

    if df.empty:
        return df

    df["prediction"] = (
        df["prediction"].astype(str)
        .str.replace("_", " ", regex=False)
        .str.title()
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


df = load_prediction_logs(PREDICTION_LOG_PATH)

if df.empty:
    st.warning("No prediction logs are available yet.")
    st.write("Logs are expected at:", f"`{PREDICTION_LOG_PATH}`")
    st.stop()

# Drop invalid timestamps so charts remain accurate.
df = df.dropna(subset=["timestamp"]).copy()
df["date"] = df["timestamp"].dt.date

latest_time = df["timestamp"].max()
recent_window = pd.Timestamp.now() - timedelta(days=7)
recent_count = df[df["timestamp"] >= recent_window].shape[0]
avg_emi = df["max_emi"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Predictions", df.shape[0])
col2.metric("Last 7 Days", recent_count)
col3.metric("Average Logged EMI", f"₹{avg_emi:.2f}" if not pd.isna(avg_emi) else "N/A")
col4.metric(
    "Most Recent Log",
    latest_time.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(latest_time) else "N/A",
)

prediction_counts = df["prediction"].value_counts()
if not prediction_counts.empty:
    top_prediction = prediction_counts.idxmax()
    top_count = int(prediction_counts.max())
    st.metric("Most Frequent Outcome", top_prediction, f"{top_count} logs")

st.divider()

st.header("Trend: Predictions per Day")
daily_trend = df.groupby("date").size().rename("count")
st.line_chart(daily_trend)

st.header("Outcome Distribution")
st.bar_chart(prediction_counts)

st.header("Recent Prediction Logs")
recent_logs = df.sort_values("timestamp", ascending=False).head(10)
st.dataframe(recent_logs, use_container_width=True)

st.markdown(
    "**Log source:** `data/prediction_logs.csv` — values are updated each time a prediction is made in the EMI predictor page."
)
