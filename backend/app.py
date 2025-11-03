from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)

# Load model and scaler
model = joblib.load("diabetes_model.joblib")
scaler = joblib.load("scaler.joblib")

# Load original dataset to get correct columns
df_train = pd.read_csv("diabetes_prediction_dataset.csv")
if "diabetes" in df_train.columns:
    df_train = df_train.drop(columns=["diabetes"])

# Encode training data to know all columns model expects
df_train_encoded = pd.get_dummies(df_train, drop_first=False)
feature_columns = df_train_encoded.columns.tolist()

# Identify numeric columns that were scaled
numeric_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Normalize key names
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

        # Convert input to DataFrame
        df_input = pd.DataFrame([data_fixed])

        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df_input, drop_first=False)
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

        # Split numeric and categorical parts
        df_numeric = df_encoded[numeric_cols]
        df_categorical = df_encoded.drop(columns=numeric_cols)

        # Scale only numeric part
        X_scaled_numeric = scaler.transform(df_numeric)

        # Combine back (keep order same as training)
        X_final = np.concatenate([X_scaled_numeric, df_categorical.values], axis=1)

        # Predict
        prediction = model.predict(X_final)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"

        return jsonify({"prediction": result})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
