import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load data
df = pd.read_csv("data/telco_churn.csv")

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
).fillna(0)

# Remove customer id
df = df.drop(columns=["customerID"])

# Convert Yes/No columns
for col in [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn"
]:
    df[col] = (df[col] == "Yes").astype(int)

# Gender
df["gender"] = (
    df["gender"] == "Male"
).astype(int)

# Label encoding
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

# Features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = GradientBoostingClassifier(
    n_estimators=200,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)

with open(
    "model/churn_model.pkl",
    "wb"
) as f:
    pickle.dump(
        {
            "model": model,
            "features": list(X.columns)
        },
        f
    )

print("Model trained successfully")
print("Saved model/churn_model.pkl")