import streamlit as st
import pandas as pd
import os
from datetime import datetime

LOG_FILE = "data/prediction_logs.csv"
ADMIN_PASSWORD = "admin123"   # 🔒 Move to settings.py later


# ----------------------------------
# SESSION STATE
# ----------------------------------
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False


# ----------------------------------
# LOGIN UI
# ----------------------------------
def admin_login():
    st.title("🔐 Admin Control Panel")

    st.markdown("Enter administrator password to continue")

    password = st.text_input("Password", type="password")

    if st.button("Login", use_container_width=True):
        if password == ADMIN_PASSWORD:
            st.session_state.admin_authenticated = True
            st.success("✅ Access Granted")
            st.rerun()
        else:
            st.error("❌ Incorrect Password")


# ----------------------------------
# LOAD LOGS
# ----------------------------------
def load_logs():
    df = pd.read_csv(
    "data/prediction_logs.csv",
    header=None,
    names=["prediction", "max_emi", "timestamp"]
    )
    return df

# ----------------------------------
# OVERVIEW TAB
# ----------------------------------
def overview_tab(df):
    st.subheader("📊 System Overview")

    col1, col2, col3 = st.columns(3)

    total = len(df)

    if not df.empty:
        eligible = (df["prediction"] == "Eligible").sum()
        high_risk = (df["prediction"] == "High Risk").sum()
        not_eligible = (df["prediction"] == "Not Eligible").sum()
    else:
        eligible = high_risk = not_eligible = 0

    col1.metric("Total Predictions", total)
    col2.metric("Eligible", eligible)
    col3.metric("High Risk", high_risk)

    st.divider()

    if not df.empty:
        st.bar_chart(df["prediction"].value_counts())


# ----------------------------------
# LOGS TAB
# ----------------------------------
def logs_tab(df):
    st.subheader("📁 Prediction Logs")

    if df.empty:
        st.info("No prediction logs available.")
        return

    filter_option = st.selectbox(
        "Filter by Prediction",
        ["All"] + list(df["prediction"].unique())
    )

    if filter_option != "All":
        df = df[df["prediction"] == filter_option]

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Filtered Logs",
        csv,
        "filtered_logs.csv",
        "text/csv",
        use_container_width=True
    )


# ----------------------------------
# MONITORING TAB
# ----------------------------------
def monitoring_tab(df):
    st.subheader("📈 Monitoring")

    if df.empty:
        st.info("No monitoring data available.")
        return

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        trend = df.groupby(df["timestamp"].dt.date)["prediction"].count()
        st.line_chart(trend)

    st.bar_chart(df["prediction"].value_counts())


# ----------------------------------
# SETTINGS TAB
# ----------------------------------
def settings_tab():
    st.subheader("⚙ Admin Settings")

    if st.button("🗑 Clear Prediction Logs", use_container_width=True):
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            st.success("Logs cleared successfully.")
        else:
            st.warning("Log file not found.")

    st.info(f"Server Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ----------------------------------
# MAIN ADMIN PANEL
# ----------------------------------
def admin_panel():

    if not st.session_state.admin_authenticated:
        admin_login()
        return

    st.sidebar.success("Admin Logged In")
    if st.sidebar.button("Logout"):
        st.session_state.admin_authenticated = False
        st.rerun()

    df = load_logs()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "📁 Logs", "📈 Monitoring", "⚙ Settings"]
    )

    with tab1:
        overview_tab(df)

    with tab2:
        logs_tab(df)

    with tab3:
        monitoring_tab(df)

    with tab4:
        settings_tab()


# Run
admin_panel()