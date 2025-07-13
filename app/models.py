import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def load_model():
    model_path = Path("saved_models/xgboost_model.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_default(model, input_data):
    # Ensure all expected columns are present
    expected_cols = ['PAY_0', 'AGE', 'BILL_AMT1', 'LIMIT_BAL', 'BILL_AMT2',
                    'SEX_2', 'EDUCATION_2', 'EDUCATION_3', 'EDUCATION_4',
                    'MARRIAGE_2', 'MARRIAGE_3']
    
    # Create DataFrame with all expected columns
    X = pd.DataFrame(columns=expected_cols)
    
    # Fill in available data
    for col in input_data.columns:
        if col in expected_cols:
            X[col] = input_data[col]
    
    # Fill missing with 0 (for one-hot encoded columns not selected)
    X.fillna(0, inplace=True)
    
    # Reorder columns to match training
    X = X[expected_cols]
    
    # Predict
    prediction = model.predict(X)
    probability = model.predict_proba(X)
    
    return prediction, probability