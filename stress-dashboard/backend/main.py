from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

# -----------------------------
# ✅ CORS FIX (CRITICAL)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend (Render static site)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_LightGBM.pkl")
model = joblib.load(MODEL_PATH)

# -----------------------------
# Request schema
# -----------------------------
class PredictRequest(BaseModel):
    features: list[float]

# -----------------------------
# Health check (used by frontend)
# -----------------------------
@app.get("/")
def health():
    return {"status": "ok"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: PredictRequest):
    X = np.array(data.features).reshape(1, -1)

    # Model prediction (assumed 0–4)
    pred = int(model.predict(X)[0])

    return {
        "stress_code": pred
    }
