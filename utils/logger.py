import pandas as pd
from datetime import datetime
from config.settings import *

def log_prediction(data):
    data["timestamp"] = datetime.now()
    df = pd.DataFrame([data])
    df.to_csv(PREDICTION_LOG_PATH, mode="a", header=False, index=False)

def log_feedback(data):
    data["timestamp"] = datetime.now()
    df = pd.DataFrame([data])
    df.to_csv(FEEDBACK_LOG_PATH, mode="a", header=False, index=False)