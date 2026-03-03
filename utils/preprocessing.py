import pandas as pd
import numpy as np
import joblib

# Define expected schema
EXPECTED_COLUMNS = [
    "age",
    "gender",
    "marital_status",
    "education",
    "monthly_salary",
    "employment_type",
    "years_of_employment",
    "company_type",
    "house_type",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "existing_loans",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "emi_scenario",
    "requested_amount",
    "requested_tenure"
]


def preprocess_input(input_dict:dict) -> pd.DataFrame:
    corrected_data_type_df = convert_to_correct_data_type(input_dict)
    skewness_corrected_df = apply_logTransform_to_columns(corrected_data_type_df)
    power_transformed_df = apply_power_transformers(skewness_corrected_df)
    df_after_one_hot = apply_one_hot_encoding(power_transformed_df)
    df_after_lebel_encode = apply_lebel_encoding(df_after_one_hot)
    df_after_ordinal_encode = apply_ordinal_encoders(df_after_lebel_encode)
    df_after_binary_enocode = apply_binary_encoding(df_after_ordinal_encode)
    final_df = create_interaction_features(df_after_binary_enocode)
   
    return final_df




def convert_to_correct_data_type(input_dict: dict) -> pd.DataFrame:
    """
    Convert raw input dictionary into
    model-ready DataFrame matching training schema.
    """

    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # Ensure correct column order
    df = df[EXPECTED_COLUMNS]

    # ---------------------------
    # Type Casting (Numeric)
    # ---------------------------
    float_cols = [
        "age", "monthly_salary", "years_of_employment",
        "monthly_rent", "school_fees", "college_fees",
        "travel_expenses", "groceries_utilities",
        "other_monthly_expenses", "current_emi_amount",
        "credit_score", "bank_balance",
        "emergency_fund", "requested_amount"
    ]

    int_cols = [
        "family_size", "dependents", "requested_tenure"
    ]

    for col in float_cols:
        df[col] = df[col].astype(np.float64)

    for col in int_cols:
        df[col] = df[col].astype(np.int64)

    # ---------------------------
    # Final Safety Check
    # ---------------------------
    if df.isnull().sum().sum() > 0:
        raise ValueError("Preprocessing Error: Null values found after encoding.")

    return df

def apply_logTransform_to_columns(data):
    df = data.copy()
    columns_to_apply_log = ['monthly_salary','years_of_employment','travel_expenses','groceries_utilities','other_monthly_expenses','current_emi_amount','requested_amount']
    for column in columns_to_apply_log:
        df[column] = np.log1p(df[[column]])
    print("completed log transform")
    return df

def apply_power_transformers(input_df: pd.DataFrame) -> pd.DataFrame:
    columns = ['monthly_rent','college_fees','bank_balance','emergency_fund']
    
    for col in columns:
        print("processing power transformscls")
        pt = joblib.load(f"D:\AI & ML\EMIPredictionApp\models\{col}_power_transformer.pkl")
        input_df[col] = pt.fit_transform(input_df[[col]])  # ONLY transform
    
    return input_df

def apply_one_hot_encoding(input_df: pd.DataFrame) -> pd.DataFrame:
    one_hot_encoding_columns = ['gender', 'marital_status', 'employment_type']
    encoder = joblib.load("D:\AI & ML\EMIPredictionApp\models\onehot_encoder.pkl")
    # Transform (NOT fit_transform)
    encoded_array = encoder.transform(input_df[one_hot_encoding_columns])

    # Convert to dataframe
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(one_hot_encoding_columns),
        index=input_df.index
    )

    # Drop original categorical columns
    input_df = input_df.drop(columns=one_hot_encoding_columns)

    # Concatenate encoded columns
    final_df = pd.concat([input_df, encoded_df], axis=1)
    print("completed one hot")

    return final_df

def apply_lebel_encoding(input_df: pd.DataFrame) -> pd.DataFrame:
    label_encoder = joblib.load("D:\AI & ML\EMIPredictionApp\models\emiscenario_lblencoder.pkl")
    # Transform (NOT fit_transform)
    df = input_df.copy()
    df['emi_scenario'] = label_encoder.transform(df['emi_scenario'])
    print("completed label encoding")
    return df

def apply_binary_encoding(input_df):
    df = input_df.copy()
    df['existing_loans'] = df['existing_loans'].map({'Yes':1,'No':0})
    print("completed binary encoding")
    return df


def apply_ordinal_encoders(input_df: pd.DataFrame) -> pd.DataFrame:
    
    # Load encoders
    education_encoder = joblib.load("D:\AI & ML\EMIPredictionApp\models\education_encoder.pkl")
    company_type_encoder = joblib.load("D:\AI & ML\EMIPredictionApp\models\company_type_encoder.pkl")
    house_type_encoder = joblib.load("D:\AI & ML\EMIPredictionApp\models\house_type_encoder.pkl")
    age_group_encoder = joblib.load("D:\AI & ML\EMIPredictionApp\models\\age_group_encoder.pkl")

    # Transform (NOT fit_transform)
    input_df['education'] = education_encoder.transform(input_df[['education']])
    input_df['company_type'] = company_type_encoder.transform(input_df[['company_type']])
    input_df['house_type'] = house_type_encoder.transform(input_df[['house_type']])
    # Define bins and labels
    bins = [25, 35, 45, 55, 65]   # edges
    age_group_order = ['25-34', '35-44', '45-54', '55-64']

    # Create age_group column
    input_df['age_group'] = pd.cut(
        input_df['age'],
        bins=bins,
        labels=age_group_order,
        right=False  # 25-34 includes 25 but excludes 35
    )

    # Make it ordered categorical (important for ML / sorting)
    input_df['age_group'] = pd.Categorical(
        input_df['age_group'],
        categories=age_group_order,
        ordered=True
    )

    input_df['age_group'] = age_group_encoder.transform(input_df[['age_group']])
    input_df.drop('age',axis=1,inplace=True)

    print("completed ordinal encoding...")

    return input_df

def create_interaction_features(df: pd.DataFrame):
    df['debt_to_income_ratio'] = df['current_emi_amount'] / df['monthly_salary']
    df['total_expenses'] = df['monthly_rent'] + df['travel_expenses'] + df['groceries_utilities'] + df['other_monthly_expenses'] + df['school_fees'] + df['college_fees']
    df['expense_to_income_ratio'] = df['total_expenses'] / df['monthly_salary']
    df['affordability_ratio'] = (df['monthly_salary'] - (df['current_emi_amount'] + df['total_expenses'])) / df['monthly_salary']

    bins_credit_score = [299, 579, 669, 739, 900]
    labels_credit_score = ['Poor', 'Fair', 'Good', 'Excellent']
    df['credit_score_category'] = pd.cut(df['credit_score'], bins=bins_credit_score, labels=labels_credit_score, right=True, include_lowest=True)
    
    credit_score_mapping = {'Poor': 3, 'Fair': 2, 'Good': 1, 'Excellent': 0}

    df['credit_score_numeric'] = df['credit_score_category'].map(credit_score_mapping).astype(int)

    df['combined_credit_risk'] = df['credit_score_numeric'] + df['existing_loans']

    df.drop('credit_score_category',axis=1,inplace=True)

    # 1. Define bins and labels for employment tenure categorization
    bins_employment = [0, 2.5, 7.5, 36]
    labels_employment = [0,1,2] #['Entry-level :0 ', 'Mid-level :1', 'Experienced:2']


    # 2. Create 'employment_tenure_category' column
    df['employment_tenure_category'] = pd.cut(df['years_of_employment'], bins=bins_employment, labels=labels_employment, right=True, include_lowest=True)
    # 3. Create 'is_long_term_employed' binary column
    df['is_long_term_employed'] = (df['years_of_employment'] >= 5).astype(int)
    

    # 1. Calculate income_per_family_member
    # Ensure family_size is not zero to avoid division by zero errors
    df['income_per_family_member'] = df['monthly_salary'] / df['family_size'].replace(0, np.nan)

    # 2. Calculate savings_to_income_ratio
    df['savings_to_income_ratio'] = (df['bank_balance'] + df['emergency_fund']) / df['monthly_salary'].replace(0, np.nan)

    # 3. Calculate credit_stability_score
    df['credit_stability_score'] = df['credit_score'] * df['years_of_employment']

    # 4. Calculate loan_affordability_index
    df['loan_affordability_index'] = df['requested_amount'] / df['monthly_salary'].replace(0, np.nan)

    df.drop(['years_of_employment'],axis=1,inplace=True)
    
    return df