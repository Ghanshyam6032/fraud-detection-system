# main.py - Advanced FastAPI Backend for Fraud Detection

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd
import joblib
import os
import logging
import uuid

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="fraud_api.log",
    filemode="a"
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# FastAPI App Initialization
# --------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="Advanced API for Credit Card Fraud Detection using Isolation Forest",
    version="2.0.0"
)

# --------------------------------------------------
# Enable CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "fraud_scale.pkl")
model = None
scaler = None

# --------------------------------------------------
# Load Model and Scaler
# --------------------------------------------------

@app.on_event("startup")
async def load_model():
    global model, scaler

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")

        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file missing: {SCALER_PATH}")

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        logger.info("Model and scaler loaded successfully")

    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise RuntimeError(f"Startup failed: {e}")

# --------------------------------------------------
# Request Model
# --------------------------------------------------

class TransactionData(BaseModel):
    amount: float = Field(..., gt=0)
    merchant_id: int = Field(..., gt=0)

# --------------------------------------------------
# Response Model
# --------------------------------------------------

class PredictionResponse(BaseModel):
    request_id: str
    status: str
    prediction_code: int
    anomaly_score: float
    confidence: float
    timestamp: str
    user_ip: str
    message: str

# --------------------------------------------------
# Root Endpoint
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "Fraud Detection API Running Successfully",
        "version": "2.0.0"
    }

# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(data: TransactionData, request: Request):

    try:
        # Unique Request ID
        request_id = str(uuid.uuid4())

        # User IP
        user_ip = request.client.host

        logger.info(f"Request ID: {request_id}")
        logger.info(f"User IP: {user_ip}")
        logger.info(f"Transaction Amount: {data.amount}")
        logger.info(f"Merchant ID: {data.merchant_id}")

        # Create DataFrame
        input_df = pd.DataFrame([{
            'Amount': data.amount,
            'MerchantID': data.merchant_id
        }])

        # Scale Input
        scaled_input = scaler.transform(
            input_df[['Amount', 'MerchantID']]
        )

        scaled_input_df = pd.DataFrame(
            scaled_input,
            columns=['Amount', 'MerchantID']
        )

        # Prediction
        prediction_result = model.predict(scaled_input_df)[0]

        # Anomaly Score
        anomaly_score = model.decision_function(
            scaled_input_df
        )[0]

        # Prediction Mapping
        status = (
            "Normal Transaction"
            if prediction_result == 1
            else "Fraudulent Transaction"
        )

        # Confidence Score
        confidence = round(abs(anomaly_score) * 100, 2)

        # Timestamp
        current_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        logger.info(
            f"Prediction Complete | "
            f"Status: {status} | "
            f"Score: {anomaly_score}"
        )

        return {
            "request_id": request_id,
            "status": status,
            "prediction_code": int(prediction_result),
            "anomaly_score": round(float(anomaly_score), 4),
            "confidence": confidence,
            "timestamp": current_time,
            "user_ip": user_ip,
            "message": "Prediction successful"
        }

    except Exception as e:

        logger.error(f"Prediction Error: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# --------------------------------------------------
# Health Check Endpoint
# --------------------------------------------------

@app.get("/health")
async def health_check():

    if model is not None and scaler is not None:

        logger.info("Health check successful")

        return {
            "status": "ok",
            "message": "API is healthy and models are loaded",
            "model_version": "IsolationForest v2.0"
        }

    else:

        logger.error("Health check failed")

        raise HTTPException(
            status_code=503,
            detail="API unhealthy: model not loaded"
        )