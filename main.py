import streamlit as st
import pandas as pd
import numpy as np
import joblib
import jsons

st.set_page_config(page_title="Cancer Prediction", layout="wide")
st.title("Cancer Prediction App")

@st.cache_data
def load_artifacts():
    artifacts = {}
    artifacts['model'] = joblib.load("cancer_model.joblib")
    artifacts['scaler'] = joblib.load("scaler.joblib")
    artifacts['imputer'] = joblib.load("imputer.joblib")
    artifacts['label_encoder'] = joblib.load("label_encoder.joblib")
    with open("feature_names.json", "r") as f:
        artifacts['features'] = json.load(f)
    return artifacts

art = load_artifacts()
features = art['features']

mode = st.sidebar.selectbox("Mode", ["Batch (CSV)", "Single sample"])

if mode == "Batch (CSV)":
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        X = df.reindex(columns=features)
        X = art['imputer'].transform(X)
        X = art['scaler'].transform(X)
        preds = art['model'].predict(X)
        preds = art['label_encoder'].inverse_transform(preds)
        df["Prediction"] = preds
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
else:
    st.header("Enter Feature Values")
    sample = {}
    for col in features:
        val = st.number_input(col, value=0.0)
        sample[col] = val
    if st.button("Predict"):
        Xs = pd.DataFrame([sample], columns=features)
        Xs = art['imputer'].transform(Xs)
        Xs = art['scaler'].transform(Xs)
        pred = art['model'].predict(Xs)
        pred = art['label_encoder'].inverse_transform(pred)
        st.success(f"Prediction: {pred[0]}")
