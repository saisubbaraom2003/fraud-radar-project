 
# scripts/train.py
import os
import mlflow
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score

# --- MLflow Setup ---
# The tracking URI is now set via an environment variable in docker-compose
mlflow.set_experiment("FraudRadar-Training")

# --- Model Definitions ---
# A selection of models to compare
models = {
    "LogisticRegression": LogisticRegression(solver='liblinear', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

def train_and_select_model(X_train, y_train, X_test, y_test):
    """
    Trains multiple models, logs their performance to MLflow, and selects the best one.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Validation data.

    Returns:
        The best performing model object and its name.
    """
    best_recall = -1
    best_model = None
    best_model_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            # Log model type
            mlflow.log_param("model_name", name)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate - for fraud, recall is often the most important metric
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            print(f"Model: {name} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
            
            # Log the model itself as an artifact
            mlflow.sklearn.log_model(model, "model")
            
            # Check if this is the best model based on recall
            if recall > best_recall:
                best_recall = recall
                best_model = model
                best_model_name = name
                
                # Tag the best run in MLflow
                mlflow.set_tag("best_model", "true")

    print(f"\n🏆 Best Model Found: {best_model_name} with Recall: {best_recall:.4f}")
    return best_model, best_model_name