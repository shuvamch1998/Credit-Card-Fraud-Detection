
🛡️ Credit Card Fraud Detection App
A full-stack machine learning system to detect fraudulent transactions and minimize financial losses for enterprises, built with XGBoost, MLflow, Docker, and Streamlit Cloud.



🚀 Project Overview
Credit card fraud causes billions in losses annually. This project aims to minimize business risk by:

Accurately detecting fraud with a high-recall ML model.

Optimizing a business-driven threshold to reduce false negatives (missed frauds).

Providing a user-friendly fraud scoring interface for real-time transactions.

✅ Live Demo: Try the deployed app on Streamlit Cloud (https://credit-card-fraud-detection-xxmir84ef2no5ccrxy4cpf.streamlit.app/)
🐳 Docker Image: Available on Docker Hub (https://hub.docker.com/repository/docker/shuvam1998/fraud-app/general)

🗂️ Dataset Information
This project uses the Credit Card Fraud Detection dataset published by Machine Learning Group – ULB, available on Kaggle.(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

📦 Size: 284,807 transactions

✅ Class 1 (Fraud): 492 instances (~0.17%)

⚠️ Extreme Class Imbalance: Required threshold optimization and careful recall tuning

🔐 Feature Masking with PCA
To protect user privacy and sensitive financial attributes:

All features (except Time and Amount) are anonymized using Principal Component Analysis (PCA).

This results in 28 features named V1 through V28 which capture the principal components of transaction behavior.

PCA preserves variance for modeling, but original feature meanings are not interpretable.

🧠 Despite this, the dataset is still extremely useful for anomaly detection, fraud modeling, and cost-sensitive ML experiments.
🧠 Model Summary
Metric	Value
Model	XGBoost
Test Set F1	0.4416 (Fraud class)
Test Set Recall	0.8878 (Fraud class)
ROC AUC	0.9788
Business Cost	💰 $13,090 (minimized on test set)
Threshold Used	0.14

⚠️ Business costs were defined as:

$1000 for a missed fraud (False Negative)

$10 for a false alarm (False Positive)

🔍 Business Impact
Objective: To help businesses reduce fraud-related losses by deploying a high-recall, cost-sensitive model.

Insight: By tuning the threshold to prioritize fraud recall, the model prevents high-value fraud losses while managing false alarms.

Result: At the optimized threshold of 0.14, the total financial loss on the test set is $13,090, down from the pre-tuned cost of $16,240 — thanks to significantly fewer missed frauds (False Negatives).

🧪 Key Features
✅ Data Pipeline: ETL with feature scaling, train-test split, and class balancing

🎯 Model Training: XGBoost with hyperparameter tuning

💼 Threshold Optimization: Custom business loss function to find optimal cutoff

📊 Evaluation: ROC Curve, Lift Curve, Confusion Matrix, SHAP 

🚀 Deployment: Dockerized ML app served via Streamlit Cloud

📂 Project Structure

├── app/
│   ├── streamlit_app.py       # Frontend UI logic
│   └── output/                # Saved model, scaler, threshold
├── notebooks/                 # EDA + model experimentation
├── src/                       # Custom Python modules
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker config for containerization
├── .gitignore
🧪 Model Tracking with MLflow

All experiments are tracked using MLflow:

Model versions

F1/Recall/Precision

Optimal thresholds

Business cost per threshold

🔎 Easily view and compare experiments with mlflow ui.

🐳 Docker Setup
Build and run locally:

docker build -t fraud-detection .
docker run -p 8501:8501 fraud-detection

Or pull directly from Docker Hub:

docker pull shuvamch1998/fraud-detection
docker run -p 8501:8501 shuvamch1998/fraud-detection
☁️ Deploy on Streamlit Cloud
Fork the repo

Add these files in the root or app/:

fraud_best_model.pkl

fraud_scaler.pkl

optimal_threshold.txt

Add streamlit_app.py as entry point

Configure requirements.txt to include:

nginx
Copy
Edit
xgboost
streamlit
scikit-learn
joblib
pandas
numpy

📌 How to Use
Upload a CSV of new transactions:

App outputs fraud probability and decision (fraud/not fraud)

Tune threshold via slider to test recall/precision trade-offs

View business cost, confusion matrix, ROC & Lift curves

📈 Results Summary
Metric	Value
Accuracy	99.61%
F1 (Fraud)	0.4416
Recall (Fraud)	0.8878
ROC AUC	0.9788

## 📉 Business Value Delivered

Without ML model: If all transactions were treated as non-fraud, business loss would be:

492 frauds × $1000 = $492,000
With the model: $13,090 → ⚡ Saved over $478,910 in test set!

This model saves nearly 97.34% of the potential fraud loss for the business.
