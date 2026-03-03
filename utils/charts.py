import plotly.express as px

def prediction_distribution_chart(df):
    fig = px.histogram(df, x="prediction", title="Prediction Distribution")
    return fig

def approval_rate_chart(df):
    fig = px.pie(df, names="prediction", title="Approval Distribution")
    return fig