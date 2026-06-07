import sys
sys.path.insert(0, ".")

import pickle
import pandas as pd
import pytest


def load_model():
    with open("model/churn_model.pkl", "rb") as f:
        return pickle.load(f)


def test_model_file_exists():
    data = load_model()
    assert "model" in data
    assert "features" in data


def test_prediction_is_valid_probability():
    data = load_model()

    model = data["model"]
    features = data["features"]

    defaults = {f: 0 for f in features}
    defaults["MonthlyCharges"] = 65.0
    defaults["TotalCharges"] = 780.0
    defaults["tenure"] = 12

    X = pd.DataFrame([defaults])[features]

    prob = float(model.predict_proba(X)[0][1])

    assert 0.0 <= prob <= 1.0


def test_high_risk_customer_has_higher_prob_than_loyal():

    data = load_model()

    model = data["model"]
    features = data["features"]

    high = {f: 0 for f in features}
    high.update({
        "tenure": 2,
        "Contract": 0,
        "MonthlyCharges": 90,
        "TotalCharges": 180
    })

    low = {f: 0 for f in features}
    low.update({
        "tenure": 72,
        "Contract": 2,
        "MonthlyCharges": 45,
        "TotalCharges": 3240
    })

    X_high = pd.DataFrame([high])[features]
    X_low = pd.DataFrame([low])[features]

    prob_high = float(model.predict_proba(X_high)[0][1])
    prob_low = float(model.predict_proba(X_low)[0][1])

    assert prob_high > prob_low


def test_prediction_is_fast():

    import time

    data = load_model()

    model = data["model"]
    features = data["features"]

    row = {f: 0 for f in features}
    row["MonthlyCharges"] = 65
    row["tenure"] = 12

    X = pd.DataFrame([row])[features]

    t0 = time.time()

    for _ in range(50):
        model.predict_proba(X)

    elapsed_ms = (time.time() - t0) * 1000 / 50

    assert elapsed_ms < 100