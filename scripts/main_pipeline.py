# scripts/main_pipeline.py
import os
import sys
import joblib
import boto3
from prefect import flow, task, get_run_logger
from dotenv import load_dotenv

# --- Path Setup ---
# Add the 'scripts' directory to Python's path to allow direct imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now you can import your modules
from data_loader import load_data
from preprocess import preprocess_data
from train import train_and_select_model
from evaluate import evaluate_best_model

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# CORRECTED PATH: Data is mounted to /app/pipeline_env/data in the container
RAW_DATA_PATH = "/app/pipeline_env/data/raw/creditcard.csv"
# CORRECTED PATH: Models are mounted to /app/pipeline_env/models in the container
MODEL_DIR = "/app/pipeline_env/models"
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# --- Prefect Tasks ---
# Each step in the pipeline is a task

@task(name="Load Data", retries=2, retry_delay_seconds=10)
def load_data_task():
    logger = get_run_logger()
    logger.info(f"Loading data from {RAW_DATA_PATH}")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {RAW_DATA_PATH}")
    return load_data(RAW_DATA_PATH, frac=0.1) # Using 10% of data for speed

@task(name="Preprocess Data")
def preprocess_data_task(df):
    logger = get_run_logger()
    logger.info("Preprocessing data: Scaling and applying SMOTE...")
    (X_train, X_test, y_train, y_test), scaler = preprocess_data(df)
    
    # Save the scaler locally first
    os.makedirs(MODEL_DIR, exist_ok=True)
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved locally to {scaler_path}")
    
    return X_train, X_test, y_train, y_test, scaler_path

@task(name="Train and Select Best Model")
def train_task(X_train, y_train, X_test, y_test):
    logger = get_run_logger()
    logger.info("Training multiple models and selecting the best one with MLflow...")
    best_model, best_model_name = train_and_select_model(X_train, y_train, X_test, y_test)
    
    # Save the best model locally
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"Best model ({best_model_name}) saved locally to {model_path}")
    
    return best_model, X_test, y_test, model_path

@task(name="Upload Artifacts to S3")
def upload_to_s3_task(local_path: str, s3_key: str):
    logger = get_run_logger()
    # Check if S3_BUCKET_NAME is set
    if not S3_BUCKET_NAME:
        logger.error("S3_BUCKET_NAME environment variable not set. Skipping S3 upload.")
        return
        
    s3 = boto3.client("s3")
    try:
        s3.upload_file(local_path, S3_BUCKET_NAME, s3_key)
        logger.info(f"✅ Successfully uploaded {local_path} to s3://{S3_BUCKET_NAME}/{s3_key}")
    except Exception as e:
        logger.error(f"❌ Failed to upload {local_path} to S3: {e}")
        # Re-raise the exception if S3 upload is critical, or just log and continue
        # For a pipeline, failing on upload might be acceptable if other steps are critical
        raise

# --- Prefect Flow ---
# The main workflow that orchestrates the tasks

@flow(name="FraudRadar Full Training Pipeline")
def training_pipeline():
    logger = get_run_logger()
    logger.info("Starting the FraudRadar training pipeline...")
    
    # 1. Load data
    df = load_data_task()
    
    # 2. Preprocess data and get scaler
    X_train, X_test, y_train, y_test, scaler_path = preprocess_data_task(df)
    
    # 3. Train models and get the best one
    best_model, X_test_eval, y_test_eval, model_path = train_task(X_train, y_train, X_test, y_test)
    
    # 4. Evaluate the final best model
    evaluate_best_model(best_model, X_test_eval, y_test_eval)
    
    # 5. Upload final artifacts to S3
    logger.info("Uploading final artifacts to S3...")
    upload_to_s3_task(scaler_path, "models/scaler.pkl")
    upload_to_s3_task(model_path, "models/best_model.pkl")
    
    # Also upload the confusion matrix logged by MLflow during training
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        upload_to_s3_task(cm_path, "reports/confusion_matrix.png")

    logger.info("🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    training_pipeline()