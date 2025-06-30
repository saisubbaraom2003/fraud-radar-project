import os
import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Define input schema (no change)
class InputData(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    def to_array(self):
        return np.array([
            self.Time, self.V1, self.V2, self.V3, self.V4, self.V5, self.V6, self.V7, self.V8,
            self.V9, self.V10, self.V11, self.V12, self.V13, self.V14, self.V15, self.V16, self.V17,
            self.V18, self.V19, self.V20, self.V21, self.V22, self.V23, self.V24, self.V25, self.V26,
            self.V27, self.V28, self.Amount
        ])

# Initialize FastAPI app (no change)
app = FastAPI()

# Enable CORS for frontend (no change)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", os.getenv("FRONTEND_URL")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the loaded model and scaler
model = None
scaler = None
model_load_status = "Not attempted"

# --- Model loading logic at startup (local file loading from new path) ---
@app.on_event("startup")
async def load_ml_artifacts_local():
    global model, scaler, model_load_status

    # Define the expected paths of the model and scaler within the container
    # Since the entire repo is copied to /usr/src/app, models/ is now at /usr/src/app/models/
    model_path = os.path.join("/usr/src/app", "models", "best_model.pkl")
    scaler_path = os.path.join("/usr/src/app", "models", "scaler.pkl")

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}.")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}.")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        model_load_status = "Model and scaler loaded successfully from local files!"
        print(model_load_status)

    except Exception as e:
        model_load_status = f"Error loading ML artifacts locally: {e}"
        print(model_load_status)
        raise RuntimeError(f"Failed to load ML artifacts locally: {e}") from e

# Basic root endpoint (updated to show model status)
@app.get("/")
def root():
    return {"message": "FraudRadar backend is running!", "model_status": model_load_status}

# Prediction endpoint (no change)
@app.post("/predict")
def predict(data: InputData):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="ML models not loaded yet.")
    
    # Ensure input is a 2D array for scaler.transform
    X = data.to_array().reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    return {"fraud": int(prediction), "probability": float(probability)}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
