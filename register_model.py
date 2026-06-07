import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


# ---------------------------------------------------
# Connect to MLflow Server
# ---------------------------------------------------

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("telco-churn")


# ---------------------------------------------------
# Load Trained Model
# ---------------------------------------------------

with open("model/churn_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
features = data["features"]


# ---------------------------------------------------
# Load Dataset Again
# Needed for metrics calculation
# ---------------------------------------------------

df = pd.read_csv("data/telco_churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
).fillna(0)

df = df.drop(columns=["customerID"])

for col in [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn"
]:
    df[col] = (df[col] == "Yes").astype(int)

df["gender"] = (df["gender"] == "Male").astype(int)

for col in [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]:
    df[col] = LabelEncoder().fit_transform(
        df[col].astype(str)
    )


# ---------------------------------------------------
# Create Test Dataset
# ---------------------------------------------------

X = df.drop(columns=["Churn"])
y = df["Churn"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------
# Calculate Metrics
# ---------------------------------------------------

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

roc_auc = round(
    roc_auc_score(y_test, y_prob),
    4
)

accuracy = round(
    accuracy_score(y_test, y_pred),
    4
)

print(f"ROC-AUC  : {roc_auc}")
print(f"Accuracy : {accuracy}")


# ---------------------------------------------------
# Register Model
# ---------------------------------------------------

with mlflow.start_run(run_name="churn-v1"):

    mlflow.log_param(
        "algorithm",
        "GradientBoostingClassifier"
    )

    mlflow.log_param(
        "version",
        "v1"
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    signature = infer_signature(
        X_test,
        y_pred
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="TelcoChurnModel"
    )

    print("\nModel Registered Successfully")
    print("Model Name : TelcoChurnModel")
    print("Version    : v1")


print("\nOpen MLflow UI")
print("http://localhost:5000")
print("Models -> TelcoChurnModel")