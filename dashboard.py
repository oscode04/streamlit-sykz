import streamlit as st
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb

# ==============================
# 1. Load model
# ==============================
with open("model_inflasi.pkl", "rb") as file:
    model = pickle.load(file)

# ==============================
# 2. Judul dashboard
# ==============================
st.title("Prediksi Inflasi - Dashboard")

# ==============================
# 3. Input user
# ==============================
st.subheader("Masukkan Data")
tahun = st.number_input("Tahun", min_value=2000, max_value=2100, value=2025)
bi_rate = st.number_input("BI Rate", value=5.75)
bbm = st.number_input("Harga BBM (Rp/liter)", value=10000)
kurs = st.number_input("Kurs USD/IDR", value=15000)
harga_beras = st.number_input("Harga Beras (Rp/kg)", value=14000)
inflasi_inti = st.number_input("Inflasi Inti (%)", value=3.0)
inflasi_total = st.number_input("Inflasi Total (%)", value=4.0)

bulan = st.slider("Bulan", 1, 12, 1)

# ==============================
# 4. Transformasi bulan ke sin & cos
# ==============================
bulan_sin = np.sin(2 * np.pi * bulan / 12)
bulan_cos = np.cos(2 * np.pi * bulan / 12)

# ==============================
# 5. Susun DataFrame sesuai urutan fitur model
# ==============================
input_data = pd.DataFrame([[
    tahun, bi_rate, bbm, kurs, harga_beras, inflasi_inti, inflasi_total, bulan_sin, bulan_cos
]], columns=[
    "Tahun", "BI_Rate", "BBM", "Kurs_USD_IDR", "Harga_Beras", "Inflasi_Inti", "Inflasi_Total", "bulan_sin", "bulan_cos"
])

# ==============================
# 6. Prediksi
# ==============================
if st.button("Prediksi"):
    # Konversi ke DMatrix agar sesuai XGBoost
    dmatrix_input = xgb.DMatrix(input_data, feature_names=input_data.columns)
    prediction = model.predict(dmatrix_input)[0]

    st.subheader("Hasil Prediksi Inflasi:")
    st.metric(label="Prediksi (%)", value=f"{prediction:.2f}")

    # Tampilkan tabel input untuk verifikasi
    st.subheader("Data yang Dimasukkan:")
    st.dataframe(input_data)
