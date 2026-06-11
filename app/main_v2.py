"""
app/main.py
===========
FastAPI prediction server for the churn model.

Endpoints:
  GET  /health      — Kubernetes liveness probe
  GET  /ready       — Kubernetes readiness probe
  GET  /metrics     — Prometheus scrape endpoint
  GET  /info        — model information
  POST /predict     — single customer prediction
  POST /predict/batch — batch predictions (up to 500 customers)

Run locally:
  uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

Test:
  curl http://localhost:8080/health
  curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"tenure": 2, "Contract": "Month-to-month", "MonthlyCharges": 85}'
"""

import os
import pickle
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

# ── Prometheus metrics ────────────────────────────────────────────
# These feed your Grafana dashboards
REQUEST_COUNT = Counter(
    "churn_requests_total",
    "Total prediction requests",
    ["status", "version"],
)
REQUEST_LATENCY = Histogram(
    "churn_latency_seconds",
    "Prediction latency in seconds",
    ["version"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)
CHURN_PROBABILITY = Histogram(
    "churn_probability_score",
    "Distribution of predicted churn probabilities",
    ["version"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── Config ────────────────────────────────────────────────────────
MODEL_PATH    = os.getenv("MODEL_PATH",    "model/churn_model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# ── Global model state ────────────────────────────────────────────
# Model is loaded once at startup, not on every request
_model_state: dict = {"model": None, "features": None, "ready": False}


def load_model():
    """Load model from disk. Called once at startup."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Run train.py first to create it."
        )
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    _model_state["model"]    = data["model"]
    _model_state["features"] = data["features"]
    _model_state["ready"]    = True
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Version: {MODEL_VERSION}")
    print(f"Features: {len(data['features'])}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load model on startup, cleanup on shutdown."""
    load_model()
    yield
    print("Server shutting down")


# ── FastAPI app ───────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="Predicts whether a telecom customer will churn",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# ── Default feature values ────────────────────────────────────────
# Used when caller does not send every field
# Values are medians from IBM Telco dataset
DEFAULTS = {
    "gender":            1,
    "SeniorCitizen":     0,
    "Partner":           0,
    "Dependents":        0,
    "tenure":            12,
    "PhoneService":      1,
    "MultipleLines":     0,
    "InternetService":   0,
    "OnlineSecurity":    0,
    "OnlineBackup":      0,
    "DeviceProtection":  0,
    "TechSupport":       0,
    "StreamingTV":       0,
    "StreamingMovies":   0,
    "Contract":          0,
    "PaperlessBilling":  1,
    "PaymentMethod":     2,
    "MonthlyCharges":    64.76,
    "TotalCharges":      2283.3,
}

# Text → number mappings (same as training)
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
INTERNET_MAP = {"DSL": 0, "Fiber optic": 1, "No": 2}
PAYMENT_MAP  = {
    "Bank transfer (automatic)":  0,
    "Credit card (automatic)":    1,
    "Electronic check":           2,
    "Mailed check":               3,
}
YESNO_MAP = {"Yes": 1, "No": 0, "yes": 1, "no": 0}


# ── Pydantic request/response models ─────────────────────────────
class CustomerFeatures(BaseModel):
    """
    Features for a single customer.
    All fields are optional — missing ones use default values.
    """
    gender:           Optional[int]   = Field(None, ge=0, le=1)
    SeniorCitizen:    Optional[int]   = Field(None, ge=0, le=1)
    Partner:          Optional[int]   = Field(None, ge=0, le=1)
    Dependents:       Optional[int]   = Field(None, ge=0, le=1)
    tenure:           Optional[int]   = Field(None, ge=0, le=240)
    PhoneService:     Optional[int]   = Field(None, ge=0, le=1)
    MultipleLines:    Optional[int]   = Field(None, ge=0, le=2)
    InternetService:  Optional[int]   = Field(None, ge=0, le=2)
    OnlineSecurity:   Optional[int]   = Field(None, ge=0, le=2)
    OnlineBackup:     Optional[int]   = Field(None, ge=0, le=2)
    DeviceProtection: Optional[int]   = Field(None, ge=0, le=2)
    TechSupport:      Optional[int]   = Field(None, ge=0, le=2)
    StreamingTV:      Optional[int]   = Field(None, ge=0, le=2)
    StreamingMovies:  Optional[int]   = Field(None, ge=0, le=2)
    Contract:         Optional[int]   = Field(None, ge=0, le=2)
    PaperlessBilling: Optional[int]   = Field(None, ge=0, le=1)
    PaymentMethod:    Optional[int]   = Field(None, ge=0, le=3)
    MonthlyCharges:   Optional[float] = Field(None, ge=0)
    TotalCharges:     Optional[float] = Field(None, ge=0)

    # Text alternatives — caller can send "Month-to-month" instead of 0
    Contract_text:        Optional[str] = Field(None, alias="Contract_text")
    InternetService_text: Optional[str] = Field(None, alias="InternetService_text")
    PaymentMethod_text:   Optional[str] = Field(None, alias="PaymentMethod_text")

    model_config = {"populate_by_name": True}


class PredictRequest(BaseModel):
    customer: CustomerFeatures
    request_id: Optional[str] = None


class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction:  int           # 0 = stays, 1 = churns
    risk_tier:         str           # LOW / MEDIUM / HIGH / CRITICAL
    model_version:     str
    latency_ms:        float
    request_id:        Optional[str] = None


class BatchPredictRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., max_length=500)
    request_id: Optional[str] = None


class BatchPredictResponse(BaseModel):
    predictions:   List[dict]
    count:         int
    model_version: str
    latency_ms:    float
    request_id:    Optional[str] = None


# ── Helper functions ──────────────────────────────────────────────
def customer_to_row(customer: CustomerFeatures) -> dict:
    """Convert a CustomerFeatures object into a feature dict the model understands."""
    row = dict(DEFAULTS)   # start with defaults

    # Override with values sent by caller
    for field, value in customer.model_dump(exclude_none=True).items():
        if value is not None and field in DEFAULTS:
            row[field] = value

    # Handle text alternatives
    if customer.Contract_text is not None:
        row["Contract"] = CONTRACT_MAP.get(customer.Contract_text, row["Contract"])
    if customer.InternetService_text is not None:
        row["InternetService"] = INTERNET_MAP.get(customer.InternetService_text, row["InternetService"])
    if customer.PaymentMethod_text is not None:
        row["PaymentMethod"] = PAYMENT_MAP.get(customer.PaymentMethod_text, row["PaymentMethod"])

    return row


def get_risk_tier(prob: float) -> str:
    if prob >= 0.75:
        return "CRITICAL"
    elif prob >= 0.50:
        return "HIGH"
    elif prob >= 0.25:
        return "MEDIUM"
    else:
        return "LOW"


def run_prediction(row: dict) -> tuple[float, int, str]:
    """Run the model on a single row dict. Returns (probability, prediction, tier)."""
    features = _model_state["features"]
    X = pd.DataFrame([row])[features]
    prob = float(_model_state["model"].predict_proba(X)[0][1])
    pred = int(prob >= 0.5)
    tier = get_risk_tier(prob)
    return prob, pred, tier


# ── Endpoints ─────────────────────────────────────────────────────
# ── Root Endpoint ────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "telco-churn-api",
        "status": "running",
        "model_version": MODEL_VERSION
    }

@app.get("/health", tags=["ops"])
def health():
    """
    Kubernetes liveness probe.
    Kubernetes calls this every 15 seconds.
    Returns 200 = pod is alive.
    Returns 500 = pod is broken, restart it.
    """
    return {
        "status":  "alive",
        "version": MODEL_VERSION,
    }


@app.get("/ready", tags=["ops"])
def ready():
    """
    Kubernetes readiness probe.
    Kubernetes waits for this to return 200 before sending traffic.
    Returns 200 = model is loaded and ready.
    Returns 503 = still loading, do not send traffic yet.
    """
    if not _model_state["ready"]:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {
        "status":  "ready",
        "version": MODEL_VERSION,
    }


@app.get("/metrics", tags=["ops"])
def metrics():
    """
    Prometheus scrape endpoint.
    Prometheus calls this every 15 seconds to collect metrics.
    Grafana reads from Prometheus to build dashboards.
    """
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/info", tags=["model"])
def info():
    """Return model metadata."""
    return {
        "model_version": MODEL_VERSION,
        "model_path":    MODEL_PATH,
        "features":      _model_state["features"],
        "ready":         _model_state["ready"],
    }


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(request: PredictRequest):
    """
    Single customer churn prediction.

    Send customer features, get back churn probability and risk tier.

    Example request:
    {
      "customer": {
        "tenure": 2,
        "Contract": 0,
        "MonthlyCharges": 85.0
      }
    }

    Or with text values:
    {
      "customer": {
        "tenure": 2,
        "Contract_text": "Month-to-month",
        "MonthlyCharges": 85.0
      }
    }
    """
    if not _model_state["ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")

    t_start = time.perf_counter()

    try:
        row  = customer_to_row(request.customer)
        prob, pred, tier = run_prediction(row)
        latency_ms = (time.perf_counter() - t_start) * 1000

        # Update Prometheus metrics
        REQUEST_COUNT.labels(status="success", version=MODEL_VERSION).inc()
        REQUEST_LATENCY.labels(version=MODEL_VERSION).observe(latency_ms / 1000)
        CHURN_PROBABILITY.labels(version=MODEL_VERSION).observe(prob)

        return PredictResponse(
            churn_probability=round(prob, 6),
            churn_prediction=pred,
            risk_tier=tier,
            model_version=MODEL_VERSION,
            latency_ms=round(latency_ms, 3),
            request_id=request.request_id,
        )

    except Exception as e:
        REQUEST_COUNT.labels(status="error", version=MODEL_VERSION).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["prediction"])
def predict_batch(request: BatchPredictRequest):
    """
    Batch predictions — up to 500 customers in one call.

    Example:
    {
      "customers": [
        {"tenure": 2, "Contract": 0},
        {"tenure": 60, "Contract": 2}
      ]
    }
    """
    if not _model_state["ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")

    t_start = time.perf_counter()

    try:
        features = _model_state["features"]
        rows = [customer_to_row(c) for c in request.customers]
        X = pd.DataFrame(rows)[features]
        probs = _model_state["model"].predict_proba(X)[:, 1]
        latency_ms = (time.perf_counter() - t_start) * 1000

        predictions = []
        for i, prob in enumerate(probs):
            prob = float(prob)
            pred = int(prob >= 0.5)
            tier = get_risk_tier(prob)
            predictions.append({
                "index":             i,
                "churn_probability": round(prob, 6),
                "churn_prediction":  pred,
                "risk_tier":         tier,
            })
            REQUEST_COUNT.labels(status="success", version=MODEL_VERSION).inc()
            CHURN_PROBABILITY.labels(version=MODEL_VERSION).observe(prob)

        REQUEST_LATENCY.labels(version=MODEL_VERSION).observe(latency_ms / 1000)

        return BatchPredictResponse(
            predictions=predictions,
            count=len(predictions),
            model_version=MODEL_VERSION,
            latency_ms=round(latency_ms, 3),
            request_id=request.request_id,
        )

    except Exception as e:
        REQUEST_COUNT.labels(status="error", version=MODEL_VERSION).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1.0/predictions")
async def seldon_predict(request: dict):

    data = request.get("data", {})
    names = data.get("names", [])
    ndarray = data.get("ndarray", [])

    if not ndarray:
        return {
            "data": {
                "names": ["prediction"],
                "ndarray": []
            }
        }

    values = ndarray[0]

    # Convert Seldon payload into dictionary
    row = dict(zip(names, values))

    # Fill missing features with model defaults
    row = {**DEFAULTS, **row}

    # Convert Contract text → numeric encoding
    if isinstance(row.get("Contract"), str):
        row["Contract"] = CONTRACT_MAP.get(
            row["Contract"],
            DEFAULTS["Contract"]
        )

    # Run prediction
    prob, pred, tier = run_prediction(row)

    return {
        "data": {
            "names": [
                "prediction",
                "probability",
                "risk_tier",
                "model_version"
            ],
            "ndarray": [[
                pred,
                round(prob, 6),
                tier,
                MODEL_VERSION
            ]]
        }
    }