# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import joblib
import json
import os

# =========================
# 1. Load Model & Data
# =========================
MODEL_PATH = "xgboost_model.json"   # model hasil save_model()
SCALER_PATH = "scaler.pkl"          # scaler (jika ada)
FEATURES_PATH = "features.pkl"      # daftar fitur (jika ada)
FEATURES_JSON = "feature_names.json" # alternatif JSON

st.set_page_config(page_title="Dashboard Prediksi Inflasi", layout="wide")

@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_scaler():
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None

@st.cache_resource
def load_features():
    if os.path.exists(FEATURES_PATH):
        return joblib.load(FEATURES_PATH)
    elif os.path.exists(FEATURES_JSON):
        with open(FEATURES_JSON, "r") as f:
            return json.load(f)
    return None

model = load_model()
scaler = load_scaler()
features = load_features()

# =========================
# 2. Input User
# =========================
st.title("📈 Prediksi Inflasi Bulanan Indonesia")
st.write("Masukkan variabel-variabel ekonomi untuk memprediksi inflasi.")

bulan_label = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]

tahun_input = st.number_input("Tahun Prediksi", min_value=2000, max_value=2100, value=datetime.now().year)
bulan_input = st.selectbox("Bulan Prediksi", bulan_label, index=6)  # default Juli

# Input variabel (HARUS sesuai urutan saat training)
BI_Rate = st.number_input("BI Rate (%)", value=6.0, step=0.1)
BBM = st.number_input("Harga BBM (Rp/liter)", value=10000, step=100)
Kurs_USD_IDR = st.number_input("Kurs USD/IDR", value=15000, step=10)
Harga_Beras = st.number_input("Harga Beras (Rp/kg)", value=12000, step=100)
Inflasi_Inti = st.number_input("Inflasi Inti (%)", value=3.0, step=0.1)
Inflasi_Total = st.number_input("Inflasi Total (%)", value=4.0, step=0.1)

# Data input
input_data = pd.DataFrame([[BI_Rate, BBM, Kurs_USD_IDR, Harga_Beras, Inflasi_Inti, Inflasi_Total]],
                          columns=['BI_Rate', 'BBM', 'Kurs_USD_IDR', 'Harga_Beras', 'Inflasi_Inti', 'Inflasi_Total'])

# Pastikan urutan fitur sama
if features:
    input_data = input_data[features]

# Scaling jika ada
if scaler:
    input_data = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

# =========================
# 3. Prediksi
# =========================
if st.button("🔮 Prediksi Inflasi"):
    dmatrix_input = xgb.DMatrix(input_data, feature_names=input_data.columns.tolist())
    y_pred = model.predict(dmatrix_input)[0]
    st.success(f"Prediksi Inflasi Bulan {bulan_input} {tahun_input}: **{y_pred:.2f}%**")

# =========================
# 4. Evaluasi Model
# =========================
st.subheader("📊 Evaluasi Model (Data Uji)")

uploaded_file = st.file_uploader("Upload Dataset Uji (CSV)", type="csv")
if uploaded_file:
    df_test = pd.read_csv(uploaded_file)

    if features:
        X_test = df_test[features]
    else:
        X_test = df_test.drop(columns=["target"], errors="ignore")

    y_test = df_test["target"]

    if scaler:
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    dtest = xgb.DMatrix(X_test, feature_names=X_test.columns.tolist())
    y_pred_test = model.predict(dtest)

    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100

    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**R²:** {r2:.4f}")
    st.write(f"**MAPE:** {mape:.2f}%")
