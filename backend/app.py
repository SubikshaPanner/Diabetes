from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import traceback
import os

app = Flask(__name__)

# ✅ Allow specific origins and handle preflight OPTIONS requests
CORS(app, resources={r"/*": {"origins": ["https://diabetes0515.vercel.app", "http://localhost:3000"]}}, supports_credentials=True)

# Load model and scaler
model = joblib.load("diabetes_model.joblib")
scaler = joblib.load("scaler.joblib")

# Load original dataset
df_train = pd.read_csv("diabetes_prediction_dataset.csv")
if "diabetes" in df_train.columns:
    df_train = df_train.drop(columns=["diabetes"])

# One-hot encode full training data to capture all columns
df_train_encoded = pd.get_dummies(df_train, drop_first=False)
feature_columns = df_train_encoded.columns.tolist()

# Columns that were scaled
numeric_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # ✅ Handle preflight OPTIONS requests explicitly
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS preflight OK"})
        response.headers.add("Access-Control-Allow-Origin", "https://diabetes0515.vercel.app")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    try:
        data = request.get_json()

        # Normalize keys
        data_fixed = {
            "age": float(data.get("age")),
            "hypertension": int(data.get("hypertension")),
            "heart_disease": int(data.get("heart_disease")),
            "bmi": float(data.get("bmi")),
            "HbA1c_level": float(data.get("hba1c_level") or data.get("HbA1c_level")),
            "blood_glucose_level": float(data.get("blood_glucose_level") or data.get("glucose")),
            "gender": data.get("gender"),
            "smoking_history": data.get("smoking_history") or data.get("smoking")
        }

        df_input = pd.DataFrame([data_fixed])

        # One-hot encode input
        df_encoded = pd.get_dummies(df_input, drop_first=False)
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

        # Split numeric + categorical
        df_numeric = df_encoded[numeric_cols]
        df_categorical = df_encoded.drop(columns=numeric_cols)

        # Scale numerical data
        X_scaled_numeric = scaler.transform(df_numeric)

        # Combine features
        X_final = np.concatenate([X_scaled_numeric, df_categorical.values], axis=1)

        # Predict
        prediction = model.predict(X_final)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"

        response = jsonify({"prediction": result})
        response.headers.add("Access-Control-Allow-Origin", "https://diabetes0515.vercel.app")
        return response

    except Exception as e:
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "https://diabetes0515.vercel.app")
        return response, 500


# ✅ Render free instance uses PORT >= 10000
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render default port
    app.run(host="0.0.0.0", port=port)
