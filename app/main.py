from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
import os
from typing import Optional

app = FastAPI()

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Configure templates and static files
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

# Load model and preprocessing objects
model_path = os.path.join(current_dir, '../saved_models/xgboost_model.pkl')
scaler_path = os.path.join(current_dir, '../saved_models/scaler.pkl')
features_path = os.path.join(current_dir, '../saved_models/top_features.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(features_path, 'rb') as f:
    top_features = pickle.load(f)

def parse_formatted_number(value: str) -> float:
    """Convert formatted string numbers (with commas) to float"""
    if isinstance(value, str):
        return float(value.replace(',', '').strip())
    return float(value)

def get_pay_status_label(value: int) -> str:
    mapping = {
        -2: "No consumption",
        -1: "Paid in full",
        0: "Revolving credit",
        1: "Payment delay for 1 month",
        2: "Payment delay for 2 months",
        3: "Payment delay for 3 months",
        4: "Payment delay for 4 months",
        5: "Payment delay for 5 months",
        6: "Payment delay for 6 months",
        7: "Payment delay for 7 months",
        8: "Payment delay for 8 months",
    }
    return mapping.get(value, "Unknown")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    PAY_0: int = Form(...),
    AGE: int = Form(...),
    BILL_AMT1: str = Form(...),  # Accept as string to handle formatting
    LIMIT_BAL: str = Form(...),   # Accept as string to handle formatting
    BILL_AMT2: str = Form(...),
    BILL_AMT3: str = Form(...),   # Accept as string to handle formatting
    MARRIAGE: str = Form(...)
):
    try:
        # Convert formatted currency strings to floats
        bill_amt1 = parse_formatted_number(BILL_AMT1)
        limit_bal = parse_formatted_number(LIMIT_BAL)
        bill_amt2 = parse_formatted_number(BILL_AMT2)
        bill_amt3 = parse_formatted_number(BILL_AMT3)

        # Prepare input data
        input_array = np.array([
            PAY_0,
            AGE,
            bill_amt1,
            limit_bal,
            bill_amt2,
            bill_amt3
        ]).reshape(1, -1).astype(float)
        
        # Scale the input
        scaled_input = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)[0][1]
        
        # Format results
        result = "High Risk" if prediction[0] == 1 else "Low Risk"
        probability_percent = round(probability * 100, 2)
        
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "result": result,
                "probability": probability_percent,
                "input_data": {
                    "PAY_0": get_pay_status_label(PAY_0),   # <-- mapped label here
                    "AGE": AGE,
                    "BILL_AMT1": bill_amt1,
                    "LIMIT_BAL": limit_bal,
                    "BILL_AMT2": bill_amt2,
                    "BILL_AMT3": bill_amt3,
                    "MARRIAGE": MARRIAGE
                }
            }
        )
    except ValueError as ve:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": f"Invalid number format: {str(ve)}"}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": f"Prediction failed: {str(e)}"}
        )
