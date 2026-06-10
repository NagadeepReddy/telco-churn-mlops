import os
import time
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from starlette.responses import Response
from pydantic import BaseModel

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    version="1.0.0"
)

# --------------------------------------------------
# Prometheus Metrics
# --------------------------------------------------

REQUEST_COUNT = Counter(
    "churn_requests_total",
    "Total prediction requests",
    ["status"]
)

REQUEST_LATENCY = Histogram(
    "churn_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

CHURN_SCORE = Histogram(
    "churn_probability",
    "Predicted churn probability",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5,
             0.6, 0.7, 0.8, 0.9, 1.0]
)

# --------------------------------------------------
# Model Loading
# --------------------------------------------------

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "model/churn_model.pkl"
)

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "v1"
)

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
features = model_data["features"]

print("Model loaded successfully")

# --------------------------------------------------
# Default Values
# --------------------------------------------------

DEFAULTS = {
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 0,
    "Dependents": 0,
    "tenure": 12,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 0,
    "OnlineSecurity": 0,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "Contract": 0,
    "PaperlessBilling": 1,
    "PaymentMethod": 2,
    "MonthlyCharges": 64.76,
    "TotalCharges": 2283.3
}

# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class PredictionRequest(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: str = "Month-to-month"

# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "service": "telco-churn-api",
        "status": "running",
        "model_version": MODEL_VERSION
    }

# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "alive",
        "model_version": MODEL_VERSION,
        "build": "cd-test-v1"
    }

# --------------------------------------------------
# Ready
# --------------------------------------------------

@app.get("/ready")
def ready():
    return {
        "status": "ready",
        "model_loaded": model is not None,
        "feature_count": len(features)
    }

# --------------------------------------------------
# Predict
# --------------------------------------------------

@app.post("/predict")
async def predict(data: PredictionRequest):

    start_time = time.time()

    try:

        raw = data.model_dump()

        row = {**DEFAULTS, **raw}

        contract_map = {
            "Month-to-month": 0,
            "One year": 1,
            "Two year": 2
        }

        if isinstance(row.get("Contract"), str):
            row["Contract"] = contract_map.get(
                row["Contract"],
                0
            )

        yes_no_map = {
            "Yes": 1,
            "No": 0
        }

        for col in [
            "Partner",
            "Dependents",
            "PhoneService",
            "PaperlessBilling"
        ]:
            if isinstance(row.get(col), str):
                row[col] = yes_no_map.get(
                    row[col],
                    0
                )

        X = pd.DataFrame([row])[features]

        probability = float(
            model.predict_proba(X)[0][1]
        )

        prediction = int(
            probability >= 0.5
        )

        if probability >= 0.75:
            risk_tier = "CRITICAL"
        elif probability >= 0.50:
            risk_tier = "HIGH"
        elif probability >= 0.25:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        latency = time.time() - start_time

        REQUEST_COUNT.labels(
            status="success"
        ).inc()

        REQUEST_LATENCY.observe(
            latency
        )

        CHURN_SCORE.observe(
            probability
        )

        return {
            "churn_probability": round(
                probability,
                4
            ),
            "churn_prediction": prediction,
            "risk_tier": risk_tier,
            "model_version": MODEL_VERSION,
            "latency_ms": round(
                latency * 1000,
                2
            )
        }

    except Exception as e:

        REQUEST_COUNT.labels(
            status="error"
        ).inc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# --------------------------------------------------
# Metrics
# --------------------------------------------------

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )