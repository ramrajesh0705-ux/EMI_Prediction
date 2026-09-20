# EMI Prediction

An interactive Streamlit application for assessing EMI affordability and loan eligibility with machine-learning models. Given a borrower’s demographic, employment, household, expense, credit, savings, and loan-request information, the application predicts an eligibility category and estimates the maximum safe monthly EMI.

## Project Overview & Purpose

`EMI_Prediction` combines two supervised-learning tasks:

1. **EMI eligibility classification** — predicts one of `Eligible`, `High_Risk`, or `Not_Eligible`.
2. **Maximum EMI regression** — estimates the borrower’s maximum safe monthly EMI in INR.

The Streamlit interface is intended for borrowers, analysts, and lending teams who want to explore the cleaned EMI dataset, test applicant scenarios, inspect prediction logs, and review basic model behavior.

The application currently loads pre-trained serialized models from `models/`; the repository does not expose a Flask/FastAPI REST service.

## Key Features

- **Interactive EMI prediction form** in `pages/3_EMI_Prediction.py`.
- **Dual-model inference:** classification confidence plus maximum monthly EMI regression.
- **Reusable preprocessing pipeline** in `utils/preprocessing.py`, including:
  - numeric type validation and schema ordering;
  - `log1p` transforms for skewed income, expense, EMI, and loan fields;
  - persisted Yeo–Johnson power transformers;
  - one-hot, ordinal, label, and binary encoding;
  - engineered affordability, debt-to-income, expense-to-income, credit-risk, savings, employment, and loan-index features.
- **Data Explorer** with row/column counts, previews, descriptive statistics, and eligibility breakdowns by gender, marital status, education, age group, house type, and company type.
- **Model Monitoring dashboard** based on `data/prediction_logs.csv`, showing prediction volume, outcome distribution, recent logs, and average logged EMI.
- **Admin Panel** for viewing/filtering/downloading logs and clearing the prediction log file.
- **Training notebooks** covering feature engineering, logistic/XGBoost classification, linear/random-forest/XGBoost regression, and evaluation visualizations.
- **Persisted model artifacts and feature-column lists** loaded with `joblib`.

## Tech Stack & Libraries

- **Language/runtime:** Python 3.9+ recommended
- **Application framework:** Streamlit 1.50.0
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn, XGBoost
- **Model persistence:** joblib / pickle artifacts
- **Visualization:** Matplotlib, Plotly, and Streamlit charts
- **Experiment artifacts:** MLflow files are included under `training/mlruns` and `mlflow.db`
- **Training environment:** Jupyter notebooks, with some notebooks originally prepared for Google Colab

The dependency versions used by the app are defined in [`requirements.txt`](requirements.txt).

## Dataset Information

The cleaned dataset is stored at [`data/emi_cleaned_data.csv`](data/emi_cleaned_data.csv). The training notebook records **404,800 rows and 27 columns**, including the following input and target fields:

### Applicant and household features

- `age`, `gender`, `marital_status`, `education`
- `family_size`, `dependents`
- `house_type`, `monthly_rent`

### Employment and income features

- `monthly_salary`
- `employment_type`, `years_of_employment`, `company_type`

### Monthly expenses and financial position

- `school_fees`, `college_fees`, `travel_expenses`
- `groceries_utilities`, `other_monthly_expenses`
- `existing_loans`, `current_emi_amount`
- `credit_score`, `bank_balance`, `emergency_fund`

### Loan request features

- `emi_scenario` — for example, personal loan, e-commerce, education, vehicle, or home-appliance EMI
- `requested_amount`
- `requested_tenure`

### Targets

- `emi_eligibility` — classification target with the categories `Eligible`, `High_Risk`, and `Not_Eligible`
- `max_monthly_emi` — regression target, expressed in INR

The training notebooks report missing values in several source columns and perform cleaning, transformations, encoding, and feature engineering before model training. The inference path expects the 26 raw input fields listed in `EXPECTED_COLUMNS` in `utils/preprocessing.py`.

## Model and Preprocessing Workflow

At inference time, the Prediction page follows this flow:

1. Collect applicant and loan details from the Streamlit form.
2. Validate column order and cast numeric fields in `convert_to_correct_data_type()`.
3. Apply log transforms and persisted power transformers.
4. Apply the saved categorical encoders.
5. Create derived affordability and credit-risk features.
6. Select the saved regression and classification training columns.
7. Run the two serialized models from `models/classification_model.pkl` and `models/regression_model.pkl`.
8. Apply `expm1` to the regression output because the regression target was log-transformed during training.
9. Display the eligibility label, class confidence, and maximum safe monthly EMI.
10. Append the result and timestamp to `data/prediction_logs.csv`.

The training notebooks explore several candidate algorithms. The checked-in application artifacts are the authoritative models used by the UI; do not assume that every notebook model is currently wired into the app.

## Repository Structure

```text
.
├── app.py                         # Streamlit entry point and global page configuration
├── pages/
│   ├── 1_Home.py                 # Project introduction and navigation guidance
│   ├── 2_Data_Explorer.py        # Dataset preview and exploratory analysis
│   ├── 3_EMI_Prediction.py       # Applicant form and model inference
│   ├── 4_Model_Monitoring.py     # Prediction-log monitoring dashboard
│   └── 5_Admin_Panel.py          # Admin login, log management, and settings
├── config/settings.py             # Model and log paths
├── data/
│   ├── emi_cleaned_data.csv      # Cleaned training/exploration dataset
│   └── prediction_logs.csv       # Append-only prediction output log
├── models/                        # Serialized models, encoders, transformers, and column lists
├── training/                      # Jupyter notebooks, evaluation plots, and MLflow artifacts
├── utils/
│   ├── preprocessing.py          # Input validation, transformations, encodings, feature engineering
│   ├── model_loader.py            # joblib model loading helpers
│   ├── logger.py                  # Prediction and feedback logging
│   ├── charts.py                  # Reusable Plotly chart helpers
│   └── theme.py                   # Streamlit styling
├── requirements.txt               # Runtime dependencies
└── README.md                      # Project documentation
```

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/ramrajesh0705-ux/EMI_Prediction.git
cd EMI_Prediction
```

### 2. Create and activate a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify required artifacts

Before starting the app, confirm that the repository contains:

- `data/emi_cleaned_data.csv`
- `models/classification_model.pkl`
- `models/regression_model.pkl`
- `models/reg_training_columns.pkl`
- `models/clf_training_columns.pkl`
- the encoder and transformer `.pkl` files referenced by `utils/preprocessing.py`

The application uses relative paths, so run it from the repository root.

## Running the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`. Use the sidebar to open:

- **Home** for the product overview;
- **Data Explorer** for dataset analysis;
- **EMI Prediction** to submit an applicant scenario;
- **Model Monitoring** to inspect prediction logs;
- **Admin Panel** to manage logs.

## Logs and Administration

- Prediction logs are written to `data/prediction_logs.csv`.
- Monitoring reads the log as three columns: prediction, maximum EMI, and timestamp.
- The Admin Panel currently contains a hard-coded `admin123` password in `pages/5_Admin_Panel.py`. **Change this before any deployment**; use environment variables or a proper identity provider instead.
- Do not commit applicant-identifying or sensitive financial data to a public repository. Add access controls and retention policies for production logs.

## Evaluation Notes

The notebooks calculate metrics appropriate to each task:

- **Regression:** MAE, MSE, R², and MAPE;
- **Classification:** accuracy, precision, recall, F1, confusion matrix, and ROC-AUC.

For example, the checked-in regression experiments report metrics for their respective notebook models and data splits. These are research artifacts, not a guarantee of production performance. Re-evaluate on a representative holdout set, check class imbalance, calibrate probabilities, and validate for fairness and drift before using predictions in lending decisions.
