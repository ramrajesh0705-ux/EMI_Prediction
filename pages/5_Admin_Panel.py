import streamlit as st
import pandas as pd
import os
from datetime import datetime

LOG_FILE = "data/prediction_logs.csv"
ADMIN_PASSWORD = "admin123"  # ⚠ Replace with env variable in production


# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


# -----------------------------
# LOGIN FUNCTION
# -----------------------------
def login():
    st.title("🔐 Admin Panel")

    password = st.text_input("Enter Admin Password", type="password")

    if st.button("Login"):
        if password == ADMIN_PASSWORD:
            st.session_state.authenticated = True
            st.success("✅ Access Granted")
            st.rerun()
        else:
            st.error("❌ Incorrect Password")


# -----------------------------
# LOAD LOG DATA
# -----------------------------
def load_logs():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        return df
    else:
        return pd.DataFrame()


# -----------------------------
# DASHBOARD - MODEL OVERVIEW
# -----------------------------
def model_overview(df):
    st.subheader("📊 Model Overview")

    col1, col2, col3 = st.columns(3)

    total_predictions = len(df)

    if not df.empty:
        high_risk = (df["prediction"] == "High Risk").sum()
        eligible = (df["prediction"] == "Eligible").sum()
        not_eligible = (df["prediction"] == "Not Eligible").sum()
    else:
        high_risk = eligible = not_eligible = 0

    col1.metric("Total Predictions", total_predictions)
    col2.metric("High Risk Count", high_risk)
    col3.metric("Eligible Count", eligible)

    st.divider()


# -----------------------------
# LOG VIEWER
# -----------------------------
def prediction_logs(df):
    st.subheader("📁 Prediction Logs")

    if df.empty:
        st.info("No prediction logs found.")
        return

    # Filter by prediction
    prediction_filter = st.selectbox(
        "Filter by Prediction",
        ["All"] + list(df["prediction"].unique())
    )

    if prediction_filter != "All":
        df = df[df["prediction"] == prediction_filter]

    st.dataframe(df, use_container_width=True)

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download CSV",
        data=csv,
        file_name="filtered_logs.csv",
        mime="text/csv",
    )


# -----------------------------
# MONITORING
# -----------------------------
def monitoring(df):
    st.subheader("📈 Monitoring Dashboard")

    if df.empty:
        st.info("No data available for monitoring.")
        return

    st.bar_chart(df["prediction"].value_counts())

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        trend = df.groupby(df["timestamp"].dt.date)["prediction"].count()
        st.line_chart(trend)


# -----------------------------
# SETTINGS
# -----------------------------
def settings():
    st.subheader("⚙ Admin Settings")

    if st.button("🗑 Clear Prediction Logs"):
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            st.success("Prediction logs cleared.")
        else:
            st.warning("No log file found.")

    st.info(f"🕒 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# -----------------------------
# MAIN ADMIN PANEL
# -----------------------------
def admin_panel():

    if not st.session_state.authenticated:
        login()
        return

    st.sidebar.success("Admin Logged In")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    df = load_logs()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "📁 Logs", "📈 Monitoring", "⚙ Settings"]
    )

    with tab1:
        model_overview(df)

    with tab2:
        prediction_logs(df)

    with tab3:
        monitoring(df)

    with tab4:
        settings()