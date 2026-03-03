import joblib
from config.settings import *

def load_classification_model():
    return joblib.load(CLASSIFICATION_MODEL_PATH)

def load_regression_model():
    return joblib.load(REGRESSION_MODEL_PATH)