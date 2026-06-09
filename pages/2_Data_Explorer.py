import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

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

# 1. EMI Eligibility by Gender
gender_eligibility_counts = df.groupby(['gender', 'emi_eligibility']).size().unstack(fill_value=0)
gender_eligibility_counts['Total_Applicants'] = gender_eligibility_counts.sum(axis=1)
gender_eligibility_counts['Approval_Percentage'] = (gender_eligibility_counts['Eligible'] / gender_eligibility_counts['Total_Applicants']) * 100
print("EMI Eligibility by Gender:")
print(gender_eligibility_counts)
print("\n")

gender_eligibility_counts[['Eligible', 'High_Risk', 'Not_Eligible']].plot(kind='bar', stacked=True, figsize=(10, 6), color=['green', 'orange', 'red'])
plt.title('EMI Eligibility Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Applicants')
plt.xticks(rotation=0)
plt.legend(title='EMI Eligibility')
plt.tight_layout()
plt.show()

# 2. EMI Eligibility by Marital Status
marital_status_eligibility_counts = df.groupby(['marital_status', 'emi_eligibility']).size().unstack(fill_value=0)
marital_status_eligibility_counts['Total_Applicants'] = marital_status_eligibility_counts.sum(axis=1)
marital_status_eligibility_counts['Approval_Percentage'] = (marital_status_eligibility_counts['Eligible'] / marital_status_eligibility_counts['Total_Applicants']) * 100
print("\nEMI Eligibility by Marital Status:")
print(marital_status_eligibility_counts)
print("\n")

marital_status_eligibility_counts[['Eligible', 'High_Risk', 'Not_Eligible']].plot(kind='bar', stacked=True, figsize=(10, 6), color=['green', 'orange', 'red'])
plt.title('EMI Eligibility Distribution by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Number of Applicants')
plt.xticks(rotation=0)
plt.legend(title='EMI Eligibility')
plt.tight_layout()
plt.show()

# 3. EMI Eligibility by Education Level
education_eligibility_counts = df.groupby(['education', 'emi_eligibility']).size().unstack(fill_value=0)
education_eligibility_counts['Total_Applicants'] = education_eligibility_counts.sum(axis=1)
education_eligibility_counts['Approval_Percentage'] = (education_eligibility_counts['Eligible'] / education_eligibility_counts['Total_Applicants']) * 100
print("\nEMI Eligibility by Education Level:")
print(education_eligibility_counts)
print("\n")

education_eligibility_counts[['Eligible', 'High_Risk', 'Not_Eligible']].plot(kind='bar', stacked=True, figsize=(10, 6), color=['green', 'orange', 'red'])
plt.title('EMI Eligibility Distribution by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Number of Applicants')
plt.xticks(rotation=45)
plt.legend(title='EMI Eligibility')
plt.tight_layout()
plt.show()

# 4. EMI Eligibility by Age Group
bins = [0, 25, 35, 45, 55, 65, np.inf]
labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

age_group_eligibility_counts = df.groupby(['age_group', 'emi_eligibility']).size().unstack(fill_value=0)
age_group_eligibility_counts['Total_Applicants'] = age_group_eligibility_counts.sum(axis=1)
age_group_eligibility_counts['Approval_Percentage'] = (age_group_eligibility_counts['Eligible'] / age_group_eligibility_counts['Total_Applicants']) * 100
print("\nEMI Eligibility by Age Group:")
print(age_group_eligibility_counts)
print("\n")

age_group_eligibility_counts[['Eligible', 'High_Risk', 'Not_Eligible']].plot(kind='bar', stacked=True, figsize=(12, 7), color=['green', 'orange', 'red'])
plt.title('EMI Eligibility Distribution by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Applicants')
plt.xticks(rotation=45)
plt.legend(title='EMI Eligibility')
plt.tight_layout()
plt.show()

# 5. EMI Eligibility by House Type
house_type_eligibility_counts = df.groupby(['house_type', 'emi_eligibility']).size().unstack(fill_value=0)
house_type_eligibility_counts['Total_Applicants'] = house_type_eligibility_counts.sum(axis=1)
house_type_eligibility_counts['Approval_Percentage'] = (house_type_eligibility_counts['Eligible'] / house_type_eligibility_counts['Total_Applicants']) * 100
print("\nEMI Eligibility by House Type:")
print(house_type_eligibility_counts)
print("\n")

house_type_eligibility_counts[['Eligible', 'High_Risk', 'Not_Eligible']].plot(kind='bar', stacked=True, figsize=(10, 6), color=['green', 'orange', 'red'])
plt.title('EMI Eligibility Distribution by House Type')
plt.xlabel('House Type')
plt.ylabel('Number of Applicants')
plt.xticks(rotation=0)
plt.legend(title='EMI Eligibility')
plt.tight_layout()
plt.show()

# 6. EMI Eligibility by Company Type
company_type_eligibility_counts = df.groupby(['company_type', 'emi_eligibility']).size().unstack(fill_value=0)
company_type_eligibility_counts['Total_Applicants'] = company_type_eligibility_counts.sum(axis=1)
company_type_eligibility_counts['Approval_Percentage'] = (company_type_eligibility_counts['Eligible'] / company_type_eligibility_counts['Total_Applicants']) * 100
print("\nEMI Eligibility by Company Type:")
print(company_type_eligibility_counts)
print("\n")

company_type_eligibility_counts[['Eligible', 'High_Risk', 'Not_Eligible']].plot(kind='bar', stacked=True, figsize=(10, 6), color=['green', 'orange', 'red'])
plt.title('EMI Eligibility Distribution by Company Type')
plt.xlabel('Company Type')
plt.ylabel('Number of Applicants')
plt.xticks(rotation=45)
plt.legend(title='EMI Eligibility')
plt.tight_layout()
plt.show()

# ==============================
# Statistical Summaries for Categorical Columns
# ==============================
categorical_columns = ['gender', 'marital_status', 'education', 'employment_type', 
                       'company_type', 'house_type', 'existing_loans', 
                       'emi_scenario', 'emi_eligibility', 'age_group']

for col in categorical_columns:
    print(f"\n--- Column: {col} ---")
    print("Value Counts:")
    print(df[col].value_counts())
    print("\nProportions:")
    print(df[col].value_counts(normalize=True))
    print("----------------------")

# ==============================
# Descriptive Statistics for Numerical Columns
# ==============================
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
descriptive_stats = df[numerical_columns].describe()
print("\nDescriptive Statistics for Numerical Columns:\n")
print(descriptive_stats)
