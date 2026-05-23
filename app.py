import shap
import pandas as pd
from flask import Flask, render_template, request
import pickle
import sqlite3

app = Flask(__name__)

model = pickle.load(open("fraud_model.pkl", "rb"))


def create_table():
    conn = sqlite3.connect("database.db")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        amount REAL,
        device_risk INTEGER,
        ip_risk INTEGER,
        result TEXT
    )
    """)

    conn.commit()
    conn.close()


create_table()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    amount = float(request.form["amount"])
    device_risk = int(request.form["device_risk"])
    ip_risk = int(request.form["ip_risk"])

    input_data = pd.DataFrame(
        [[amount, device_risk, ip_risk]],
        columns=["amount", "device_risk", "ip_risk"]
    )

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]
    probability_percent = round(probability * 100, 2)

    risk_score = (device_risk + ip_risk) / 2

    if prediction[0] == 1:
        result = "Fraud Transaction Detected"
    else:
        result = "Safe Transaction"

    if amount > 5000:
        reason = "High transaction amount increased the fraud risk."
    elif device_risk >= 7:
        reason = "High device risk score contributed to the fraud prediction."
    elif ip_risk >= 7:
        reason = "High IP risk score contributed to the fraud prediction."
    else:
        reason = "Transaction appears normal based on amount, device risk, and IP risk."

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    feature_names = ["Amount", "Device Risk", "IP Risk"]

    feature_impacts = {}

    for i, feature in enumerate(feature_names):
        feature_impacts[feature] = round(
            abs(shap_values[0][i]),
            3
        )

    top_feature = max(
        feature_impacts,
        key=feature_impacts.get
    )

    shap_reason = f"{top_feature} had the highest impact on this prediction."

    conn = sqlite3.connect("database.db")

    conn.execute(
        "INSERT INTO predictions (amount, device_risk, ip_risk, result) VALUES (?, ?, ?, ?)",
        (amount, device_risk, ip_risk, result)
    )

    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        result=result,
        reason=reason,
        risk_score=risk_score,
        probability_percent=probability_percent,
        shap_reason=shap_reason
    )


@app.route("/history")
def history():

    conn = sqlite3.connect("database.db")

    predictions = conn.execute(
        "SELECT * FROM predictions"
    ).fetchall()

    conn.close()

    return render_template(
        "history.html",
        predictions=predictions
    )


if __name__ == "__main__":
    app.run(debug=True)