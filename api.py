from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

model = pickle.load(open("fraud_model.pkl", "rb"))


class Transaction(BaseModel):
    amount: float
    device_risk: int
    ip_risk: int


@app.get("/")
def home():
    return {"message": "API working"}


@app.post("/predict")
def predict(transaction: Transaction):

    data = [[
        transaction.amount,
        transaction.device_risk,
        transaction.ip_risk
    ]]

    prediction = model.predict(data)[0]
    probability = float(model.predict_proba(data)[0][1])

    if prediction == 1:
        result = "Fraud Transaction Detected"
    else:
        result = "Safe Transaction"

    return {
    "result": str(result),
    "fraud_probability": float(round(probability * 100, 2))
}