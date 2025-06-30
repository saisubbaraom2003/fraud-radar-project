import os
import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow # Import MLflow
import mlflow.pyfunc # If your model is logged as a pyfunc model

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
    allow_origins=["http://localhost:3000", os.getenv("FRONTEND_URL")], # Added FRONTEND_URL for Render
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the loaded model and scaler
model = None
scaler = None
model_load_status = "Not attempted"

# --- Model loading logic at startup ---
@app.on_event("startup")
async def load_ml_artifacts():
    global model, scaler, model_load_status

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_artifact_bucket_name = os.getenv("MLFLOW_ARTIFACT_BUCKET_NAME") # Make sure this is also passed to backend service
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID") # These are used by boto3 internally by mlflow
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY") # These are used by boto3 internally by mlflow

    if not all([mlflow_tracking_uri, mlflow_artifact_bucket_name, aws_access_key_id, aws_secret_access_key]):
        model_load_status = "Error: Missing MLflow/AWS environment variables. Cannot load model."
        print(model_load_status)
        # Raising an exception here will cause the service to fail to start
        raise RuntimeError(model_load_status)

    print(f"MLflow Tracking URI: {mlflow_tracking_uri}")
    print(f"MLflow Artifact Bucket Name: {mlflow_artifact_bucket_name}")

    # Set MLflow tracking URI for the client
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Set AWS credentials for MLflow's S3 access (mlflow uses boto3, which reads these)
    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key

    # Define the registered model name you used in your training pipeline
    # IMPORTANT: Adjust "FraudDetectionModel" if your model is registered under a different name
    registered_model_name = "FraudDetectionModel" # Example, replace with your actual model name
    
    # Path where MLflow will download artifacts (use /tmp for temporary storage on Render)
    local_download_path = "/tmp/mlflow_artifacts"
    os.makedirs(local_download_path, exist_ok=True)


    try:
        # Load the latest version of the registered model
        # This will download the model's artifacts to a local temp directory
        # If your model was logged using `mlflow.sklearn.log_model` or `mlflow.pyfunc.log_model`
        # and registered, this is the recommended way to load.
        # It handles extracting the .pkl or other components.

        # mlflow.pyfunc.load_model expects the full URI to the MLmodel directory
        # If your model is structured: registered_model_name/MLmodel
        model_uri = f"models:/{registered_model_name}/latest"
        print(f"Attempting to load model from MLflow URI: {model_uri}")
        
        # This will download the model directory, then load the pyfunc model
        model = mlflow.pyfunc.load_model(model_uri)
        
        # --- Now for the scaler, which might be a separate artifact ---
        # You'll need to know the run_id that logged your scaler, or register the scaler too.
        # For simplicity, if the scaler is a separate artifact associated with the SAME model run,
        # you can try to download it from the same model's run artifacts.
        # A more robust solution might be to register the scaler as a separate MLflow artifact
        # or include it within the pyfunc model's artifacts.

        # For now, let's assume the scaler is also logged as a simple artifact
        # within the run that produced the latest registered model.
        # This part is more speculative and might need adjustment based on your training script.
        
        # Option 1: If scaler is logged within the model's main artifact directory
        # e.g., if you log model dir, and scaler.pkl is inside it.
        # When `mlflow.pyfunc.load_model` downloads, it gets the entire model folder.
        # You can then find the scaler within `model.metadata.saved_model_path` (local path)
        
        # A safer bet for now is to rely on a known artifact path or register the scaler.
        # For this example, let's assume the scaler is within the same `models:/` URI after download.
        
        # To download specific artifact *files* from a registered model's run:
        # First, get the run_id for the latest version of the registered model
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(name=registered_model_name, stages=["Production", "Staging", "None"])[0]
        run_id = latest_version.run_id
        
        print(f"Loading scaler from run_id: {run_id}")
        
        # Download the scaler.pkl directly to the local_download_path
        # IMPORTANT: 'scaler.pkl' here is the artifact_path when you LOGGED the scaler in Prefect/MLflow.
        # Adjust if your artifact path for the scaler is different (e.g., 'artifacts/scaler.pkl')
        scaler_artifact_path = "scaler.pkl" # This is the path *within the MLflow run's artifact URI*
                                           # where your scaler was logged.
                                           # e.g., if you did `mlflow.log_artifact("scaler.pkl")`
        
        local_scaler_file_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/{scaler_artifact_path}",
            dst_path=local_download_path
        )
        
        print(f"Scaler downloaded to: {local_scaler_file_path}")
        
        # Load the scaler using joblib
        scaler = joblib.load(local_scaler_file_path)

        model_load_status = "Model and scaler loaded successfully from MLflow!"
        print(model_load_status)

    except Exception as e:
        model_load_status = f"Error loading ML artifacts: {e}"
        print(model_load_status)
        # Re-raise for server startup to fail if model is critical
        raise RuntimeError(f"Failed to load ML artifacts: {e}") from e

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
    probability = model.predict_proba(X_scaled)[0][1] # Assuming binary classification, get probability of positive class
    return {"fraud": int(prediction), "probability": float(probability)}

if __name__ == "__main__":
    # Ensure this matches the CMD in your Dockerfile for local development if needed
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) # Changed host to 0.0.0.0 for Docker compatibility
