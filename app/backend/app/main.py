import os
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


# Define the absolute path where models are expected inside the Docker container
# The docker-compose.yaml mounts ./models (host) to /app/models (container)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Define input schema
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

# Initialize app
app = FastAPI()

# Enable CORS for frontend (adjust if frontend is hosted differently)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Construct full paths to the model and scaler files
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Path to actual model directory (not inside /backend/app/)
MODEL_DIR = os.path.join(BASE_DIR, "models")
model_path = os.path.join(MODEL_DIR, "best_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

# Load trained model and scaler
# ✅ Defensive loading with error check
if not os.path.exists(model_path):
    raise RuntimeError(f"Model file not found at {model_path}. Please run training pipeline first.")
if not os.path.exists(scaler_path):
    raise RuntimeError(f"Scaler file not found at {scaler_path}. Please run training pipeline first.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.get("/")
def root():
    return {"message": "FraudRadar backend is running!"}

@app.post("/predict")
def predict(data: InputData):
    X = data.to_array().reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    return {"fraud": int(prediction), "probability": float(probability)}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

