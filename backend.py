# backend.py – FINAL, BELIEVABLE HEAT STROKE RISK MODEL
# Hybrid Clinical Rules + Neural Network

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TensorFlow logs

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import joblib

# ================= CONFIG =================

DATA_PATH = "augmented_dataset (1).xlsx"
TARGET_COL = "Heat stroke"

MODEL_PATH = "heat_nn_model.keras"
SCALER_PATH = "scaler.pkl"

FEATURE_COLS = [
    "Patient temperature",
    "Heat Index (HI)",
    "Strenuous exercise",
    "Age",
    "Rectal temperature (deg C)",
    "Systolic BP",
    "Diastolic BP"
]

# ================= LOAD DATA =================

data = pd.read_excel(DATA_PATH)

le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object" and col != TARGET_COL:
        data[col] = le.fit_transform(data[col].astype(str))

X = data[FEATURE_COLS].values
y = data[TARGET_COL].round().astype(int).values

# Means for optional inputs
feature_means = data[FEATURE_COLS].mean()

# ================= TRAIN / LOAD NN =================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    model = Sequential([
        Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

# ================= FLASK =================

app = Flask(__name__)
CORS(app)

# ================= CLINICAL RULE ENGINE =================

def clinical_rule_score(body_temp, heat_index, rectal_temp, exercise, age):
    """
    Returns a risk score between 0 and 1 based on medical thresholds.
    Rules dominate ML in dangerous conditions.
    """
    score = 0.0

    # Core / rectal temperature (MOST IMPORTANT)
    if rectal_temp >= 41.0:
        return 0.95
    elif rectal_temp >= 40.0:
        score += 0.60
    elif rectal_temp >= 39.0:
        score += 0.45
    elif rectal_temp >= 38.0:
        score += 0.30

    # Heat index contribution
    if heat_index >= 54:
        score += 0.35
    elif heat_index >= 46:
        score += 0.25
    elif heat_index >= 40:
        score += 0.18
    elif heat_index >= 32:
        score += 0.10

    # Physical exertion
    if exercise == 1:
        score += 0.12

    # Age vulnerability
    if age >= 60:
        score += 0.10
    elif age <= 12:
        score += 0.08

    return min(score, 0.85)

# ================= PREDICTION ENDPOINT =================

@app.route("/predict", methods=["POST"])
def predict():
    data_json = request.get_json(force=True)

    def get_value(key, col):
        val = data_json.get(key)
        return float(val) if val not in [None, ""] else feature_means[col]

    body_temp   = get_value("body_temp", "Patient temperature")
    heat_index  = get_value("heat_index", "Heat Index (HI)")
    exercise    = int(get_value("exercise", "Strenuous exercise"))
    age         = get_value("age", "Age")
    rectal_temp = get_value("rectal_temp", "Rectal temperature (deg C)")
    sys_bp      = get_value("sys_bp", "Systolic BP")
    dia_bp      = get_value("dia_bp", "Diastolic BP")

    # Neural Network probability
    X_input = np.array([[body_temp, heat_index, exercise, age,
                          rectal_temp, sys_bp, dia_bp]])
    X_scaled = scaler.transform(X_input)
    nn_prob = float(model.predict(X_scaled, verbose=0)[0][0])

    # Rule-based probability
    rule_prob = clinical_rule_score(
        body_temp, heat_index, rectal_temp, exercise, age
    )

    # ================= FINAL HYBRID LOGIC =================
    # Rules DOMINATE when danger is detected

    if rule_prob >= 0.60:
        final_prob = max(rule_prob, nn_prob)
    elif rule_prob >= 0.40:
        final_prob = 0.7 * rule_prob + 0.3 * nn_prob
    else:
        final_prob = 0.6 * nn_prob + 0.4 * rule_prob

    final_prob = max(0.01, min(final_prob, 0.99))

    # Risk buckets
    if final_prob < 0.35:
        risk = "Low"
    elif final_prob < 0.65:
        risk = "Moderate"
    else:
        risk = "High"

    return jsonify({
        "probability": round(final_prob, 4),
        "risk_level": risk
    })

# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True)
