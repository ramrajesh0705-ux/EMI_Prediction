import streamlit as st
import pandas as pd
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

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
if not missing_values.empty:
    st.subheader("Missing values")
    st.dataframe(missing_values.to_frame("missing_count"))
else:
    st.subheader("Missing values")
    st.success("No missing values detected in the cleaned dataset.")

st.subheader("Numeric feature summary")
st.dataframe(df.select_dtypes(include=["number"]).describe().T)

st.header("Target and categorical distributions")

eligibility_counts = df["emi_eligibility"].value_counts().reset_index()
eligibility_counts.columns = ["emi_eligibility", "count"]
fig_eligibility = px.bar(
    eligibility_counts,
    x="emi_eligibility",
    y="count",
    color="emi_eligibility",
    title="EMI Eligibility Distribution",
    text="count",
)
fig_eligibility.update_layout(showlegend=False)
st.plotly_chart(fig_eligibility, use_container_width=True)

category_fields = ["gender", "marital_status", "education", "house_type", "age_group"]
for field in category_fields:
    if field in df.columns:
        counts = df[field].value_counts().reset_index()
        counts.columns = [field, "count"]
        fig = px.bar(
            counts,
            x=field,
            y="count",
            color=field,
            title=f"Distribution of {field.replace('_', ' ').title()}",
            text="count",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

st.header("Scenario and approval analysis")
scenario_counts = (
    df.groupby(["emi_scenario", "emi_eligibility"]).size().reset_index(name="count")
)
fig_scenario = px.bar(
    scenario_counts,
    x="emi_scenario",
    y="count",
    color="emi_eligibility",
    title="EMI Scenario vs Eligibility",
    barmode="stack",
    category_orders={"emi_eligibility": ["Eligible", "High_Risk", "Not_Eligible"]},
)
fig_scenario.update_xaxes(tickangle=-45)
st.plotly_chart(fig_scenario, use_container_width=True)

approval_rate = (
    df.assign(
        approved=df["emi_eligibility"].map({"Eligible": 1, "High_Risk": 0, "Not_Eligible": 0})
    )
    .groupby("emi_scenario", as_index=False)["approved"]
    .mean()
)
fig_approval = px.line(
    approval_rate,
    x="emi_scenario",
    y="approved",
    markers=True,
    title="EMI Approval Rate by Scenario",
)
fig_approval.update_layout(yaxis=dict(tickformat=".0%", range=[0, 1]))
fig_approval.update_xaxes(tickangle=-45)
st.plotly_chart(fig_approval, use_container_width=True)

eligible_df = df[df["emi_eligibility"] == "Eligible"]
if not eligible_df.empty:
    fig_box = px.box(
        eligible_df,
        x="emi_scenario",
        y="max_monthly_emi",
        title="Max Monthly EMI for Eligible Applicants by Scenario",
    )
    fig_box.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_box, use_container_width=True)

st.header("Numeric correlation")
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix for Numeric Features",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    st.plotly_chart(fig_corr, use_container_width=True)
