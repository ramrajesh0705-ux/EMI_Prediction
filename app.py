import streamlit as st
from utils.theme import apply_theme

st.set_page_config(
    page_title="FinTech ML Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()

st.title("💼 FinTech ML Prediction Platform")
st.markdown("AI-powered EMI Risk & Approval Prediction System")