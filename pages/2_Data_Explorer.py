import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("📊 EMI Data Explorer")
st.markdown(
    "This page shows exploratory analysis for the cleaned EMI dataset stored in `data/emi_cleaned_data.csv`."
)

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("data/emi_cleaned_data.csv")

# Load cleaned EMI dataset
df = load_data()

st.header("Dataset overview")
col1, col2 = st.columns(2)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])

st.markdown("**Preview of the first 10 records**")
st.dataframe(df.head(10))

# Helper function to plot stacked bar charts
def plot_stacked_bar(data, columns, title, xlabel, ylabel, rot=0):
    fig, ax = plt.subplots(figsize=(10, 6))
    data[columns].plot(kind='bar', stacked=True, color=['green', 'orange', 'red'], ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rot)
    ax.legend(title='EMI Eligibility')
    plt.tight_layout()
    st.pyplot(fig)

# 1. EMI Eligibility by Gender
gender_eligibility_counts = df.groupby(['gender', 'emi_eligibility']).size().unstack(fill_value=0)
gender_eligibility_counts['Total_Applicants'] = gender_eligibility_counts.sum(axis=1)
gender_eligibility_counts['Approval_Percentage'] = (gender_eligibility_counts['Eligible'] / gender_eligibility_counts['Total_Applicants']) * 100

st.subheader("EMI Eligibility by Gender")
st.dataframe(gender_eligibility_counts)
plot_stacked_bar(gender_eligibility_counts, ['Eligible', 'High_Risk', 'Not_Eligible'],
                 'EMI Eligibility Distribution by Gender', 'Gender', 'Number of Applicants', rot=0)

# 2. EMI Eligibility by Marital Status
marital_status_eligibility_counts = df.groupby(['marital_status', 'emi_eligibility']).size().unstack(fill_value=0)
marital_status_eligibility_counts['Total_Applicants'] = marital_status_eligibility_counts.sum(axis=1)
marital_status_eligibility_counts['Approval_Percentage'] = (marital_status_eligibility_counts['Eligible'] / marital_status_eligibility_counts['Total_Applicants']) * 100

st.subheader("EMI Eligibility by Marital Status")
st.dataframe(marital_status_eligibility_counts)
plot_stacked_bar(marital_status_eligibility_counts, ['Eligible', 'High_Risk', 'Not_Eligible'],
                 'EMI Eligibility Distribution by Marital Status', 'Marital Status', 'Number of Applicants', rot=0)

# 3. EMI Eligibility by Education Level
education_eligibility_counts = df.groupby(['education', 'emi_eligibility']).size().unstack(fill_value=0)
education_eligibility_counts['Total_Applicants'] = education_eligibility_counts.sum(axis=1)
education_eligibility_counts['Approval_Percentage'] = (education_eligibility_counts['Eligible'] / education_eligibility_counts['Total_Applicants']) * 100

st.subheader("EMI Eligibility by Education Level")
st.dataframe(education_eligibility_counts)
plot_stacked_bar(education_eligibility_counts, ['Eligible', 'High_Risk', 'Not_Eligible'],
                 'EMI Eligibility Distribution by Education Level', 'Education Level', 'Number of Applicants', rot=45)

# 4. EMI Eligibility by Age Group
bins = [0, 25, 35, 45, 55, 65, np.inf]
labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

age_group_eligibility_counts = df.groupby(['age_group', 'emi_eligibility']).size().unstack(fill_value=0)
age_group_eligibility_counts['Total_Applicants'] = age_group_eligibility_counts.sum(axis=1)
age_group_eligibility_counts['Approval_Percentage'] = (age_group_eligibility_counts['Eligible'] / age_group_eligibility_counts['Total_Applicants']) * 100

st.subheader("EMI Eligibility by Age Group")
st.dataframe(age_group_eligibility_counts)
plot_stacked_bar(age_group_eligibility_counts, ['Eligible', 'High_Risk', 'Not_Eligible'],
                 'EMI Eligibility Distribution by Age Group', 'Age Group', 'Number of Applicants', rot=45)

# 5. EMI Eligibility by House Type
house_type_eligibility_counts = df.groupby(['house_type', 'emi_eligibility']).size().unstack(fill_value=0)
house_type_eligibility_counts['Total_Applicants'] = house_type_eligibility_counts.sum(axis=1)
house_type_eligibility_counts['Approval_Percentage'] = (house_type_eligibility_counts['Eligible'] / house_type_eligibility_counts['Total_Applicants']) * 100

st.subheader("EMI Eligibility by House Type")
st.dataframe(house_type_eligibility_counts)
plot_stacked_bar(house_type_eligibility_counts, ['Eligible', 'High_Risk', 'Not_Eligible'],
                 'EMI Eligibility Distribution by House Type', 'House Type', 'Number of Applicants', rot=0)

# 6. EMI Eligibility by Company Type
company_type_eligibility_counts = df.groupby(['company_type', 'emi_eligibility']).size().unstack(fill_value=0)
company_type_eligibility_counts['Total_Applicants'] = company_type_eligibility_counts.sum(axis=1)
company_type_eligibility_counts['Approval_Percentage'] = (company_type_eligibility_counts['Eligible'] / company_type_eligibility_counts['Total_Applicants']) * 100

st.subheader("EMI Eligibility by Company Type")
st.dataframe(company_type_eligibility_counts)
plot_stacked_bar(company_type_eligibility_counts, ['Eligible', 'High_Risk', 'Not_Eligible'],
                 'EMI Eligibility Distribution by Company Type', 'Company Type', 'Number of Applicants', rot=45)

# ==============================
# Statistical Summaries for Categorical Columns
# ==============================
st.header("Statistical Summaries")
st.subheader("Categorical Columns")
categorical_columns = ['gender', 'marital_status', 'education', 'employment_type', 
                       'company_type', 'house_type', 'existing_loans', 
                       'emi_scenario', 'emi_eligibility', 'age_group']

for col in categorical_columns:
    st.markdown(f"**Column: {col}**")
    st.write("Value Counts:")
    st.dataframe(df[col].value_counts().reset_index().rename(columns={'index': col, col: 'count'}))
    st.write("Proportions:")
    st.dataframe(df[col].value_counts(normalize=True).reset_index().rename(columns={'index': col, col: 'proportion'}))
    st.markdown("---")

# ==============================
# Descriptive Statistics for Numerical Columns
# ==============================
st.subheader("Descriptive Statistics for Numerical Columns")
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
descriptive_stats = df[numerical_columns].describe()
st.dataframe(descriptive_stats)
