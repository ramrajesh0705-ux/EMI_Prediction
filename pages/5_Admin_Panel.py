import streamlit as st
import pandas as pd

st.title("🔐 Admin Panel")

password = st.text_input("Enter Admin Password", type="password")

if password == "admin123":
    st.success("Access Granted")

    if st.button("Clear Logs"):
        open("data/prediction_logs.csv", "w").close()
        st.warning("Logs Cleared")
else:
    st.error("Access Denied")