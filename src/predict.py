import os
import joblib
import pandas as pd
import numpy as np

# ------------- Paths -------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # .../SOP/src
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model_LightGBM.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.joblib")


def load_artifacts():
    """Load trained model, scaler and feature names."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, scaler, feature_names


def predict(input_dict):
    """
    Predict stress class from a single sample.

    input_dict: dict mapping feature_name -> value
    """
    model, scaler, feature_names = load_artifacts()

    # Make DataFrame and ensure column order is same as training
    df = pd.DataFrame([input_dict])
    df = df[feature_names]          # reorder / select columns

    X = scaler.transform(df)
    y_pred = model.predict(X)[0]
    return y_pred

if __name__ == "__main__":
    _, _, feature_names = load_artifacts()
    print("Expected feature names (in order):")
    print(feature_names)

    # Example sample – put realistic values here
    sample = {
        "SDRR": 120.0,
        "RMSSD": 35.0,
        "KURT": 2.1,
        "SKEW": 0.3,
        "MEAN_REL_RR": 0.98,
        "MEDIAN_REL_RR": 1.01,
        "SDRR_RMSSD_REL_RR": 0.75,
        "LF_NU": 55.0,
        "sampen": 1.2,
    }

    pred = predict(sample)
    print("\nPredicted stress level:", pred)
