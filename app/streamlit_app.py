import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")
st.title("🛡️ Credit Card Fraud Detection App")


MODEL_PATH = "output/fraud_best_model.pkl"
SCALER_PATH = "output/fraud_scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


uploaded_file = st.file_uploader("Upload transaction data CSV", type=["csv"])


threshold = st.slider("Select Decision Threshold", min_value=0.01, max_value=1.0, value=0.5, step=0.01)


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if 'Time' in df.columns:
            df = df.drop('Time', axis=1)
        
        
        if 'Class' in df.columns:
            df = df.drop('Class', axis=1)

        
        X_scaled = scaler.transform(df)

        
        probs = model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= threshold).astype(int)

        
        result_df = pd.DataFrame({
            "probability": np.round(probs, 6),
            "prediction": preds
        })

        st.markdown("### 🔍 Prediction Results")
        st.dataframe(result_df)

        
        result_csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", data=result_csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"🚨 Error: {str(e)}")
