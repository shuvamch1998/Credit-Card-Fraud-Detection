import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np

# === Step 1: Resolve absolute paths ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

MODEL_PATH = os.path.join(OUTPUT_DIR, "fraud_best_model.pkl")
THRESHOLD_PATH = os.path.join(OUTPUT_DIR, "optimal_threshold.txt")
XTEST_PATH = os.path.join(OUTPUT_DIR, "X_test_scaled.pkl")
YTEST_PATH = os.path.join(OUTPUT_DIR, "y_test.pkl")

# === Step 2: Load model and threshold ===
best_model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    optimal_threshold = float(f.read())

# === Step 3: Load test data ===
X_test_scaled = joblib.load(XTEST_PATH)
y_test = joblib.load(YTEST_PATH)

# === Step 4: Predict and apply threshold ===
y_scores = best_model.predict_proba(X_test_scaled)[:, 1]
y_pred_thresh = (y_scores >= optimal_threshold).astype(int)

# === Step 5: Compute evaluation metrics ===
report = classification_report(y_test, y_pred_thresh, output_dict=True)
final_recall = report["1"]["recall"]
final_precision = report["1"]["precision"]
final_f1 = report["1"]["f1-score"]
final_roc_auc = roc_auc_score(y_test, y_scores)

final_cm = confusion_matrix(y_test, y_pred_thresh)
FP = final_cm[0][1]
FN = final_cm[1][0]
cost_fn = 1000
cost_fp = 10
final_cost = FN * cost_fn + FP * cost_fp

# === Step 6: Log metrics and artifacts to MLflow ===
mlflow.set_experiment("CreditCardFraudDetection")

with mlflow.start_run(run_name="fraud-detection-v1"):
    # Log hyperparameters
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("cost_fn", cost_fn)
    mlflow.log_param("cost_fp", cost_fp)
    mlflow.log_param("optimal_threshold", optimal_threshold)

    # Log evaluation metrics
    mlflow.log_metric("recall_fraud", final_recall)
    mlflow.log_metric("precision_fraud", final_precision)
    mlflow.log_metric("f1_fraud", final_f1)
    mlflow.log_metric("roc_auc", final_roc_auc)
    mlflow.log_metric("final_business_cost", final_cost)

    # Log model and artifacts
    mlflow.sklearn.log_model(best_model, artifact_path="fraud_model")
    mlflow.log_artifact(THRESHOLD_PATH)
    mlflow.log_artifact(MODEL_PATH)
