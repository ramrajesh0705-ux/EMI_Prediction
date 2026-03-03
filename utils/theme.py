import streamlit as st

def apply_theme():
    st.markdown("""
    <style>
    body {
        background-color: #0f172a;
        color: white;
    }
    .stButton>button {
        background-color: #14b8a6;
        color: white;
        border-radius: 10px;
        width: 100%;
    }
    .stTextInput, .stNumberInput {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)