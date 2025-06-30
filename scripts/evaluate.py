 
# scripts/evaluate.py
import os
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def evaluate_best_model(model, X_test, y_test):
    """
    Performs a final, detailed evaluation of the best model and logs artifacts.

    Args:
        model: The trained model object.
        X_test, y_test: The test dataset.
    """
    # Start a new "final evaluation" run in MLflow
    with mlflow.start_run(run_name="Final_Model_Evaluation", nested=True):
        mlflow.set_tag("model_name", model.__class__.__name__)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Log main metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("final_recall", report['1']['recall'])
        mlflow.log_metric("final_precision", report['1']['precision'])
        mlflow.log_metric("final_f1_score", report['1']['f1-score'])
        mlflow.log_metric("final_auc", roc_auc_score(y_test, y_proba))
        
        print("\n--- Final Model Evaluation ---")
        print(classification_report(y_test, y_pred))

        # Create and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Fraud', 'Fraud'],
                    yticklabels=['Not Fraud', 'Fraud'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Final Confusion Matrix")
        
        # Save figure locally before logging
        os.makedirs("/app/models", exist_ok=True)
        cm_path = "/app/models/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        
        # Log artifact to MLflow
        mlflow.log_artifact(cm_path, "evaluation_plots")
        print("Final evaluation metrics and confusion matrix logged to MLflow.")