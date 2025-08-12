import streamlit as st
from src.preprocessing import preprocess_and_update_histori
from src.inference import predict_inflasi
import pandas as pd
from datetime import datetime

# Load fitur training yang sudah lengkap dan urut
with open('data/features_training.txt') as f:
    features_training = [line.strip() for line in f.readlines()]

st.title("📈 Prediksi Inflasi - Dashboard Forecasting")

st.markdown("""
Masukkan data ekonomi **bulan sebelumnya** untuk memprediksi **inflasi bulan berikutnya**.
""")

# ===== Dropdown untuk memilih bulan dan tahun =====
st.subheader("📆 Pilih Bulan & Tahun Prediksi")

# List nama bulan
bulan_dict = {
    "Januari": 1, "Februari": 2, "Maret": 3, "April": 4,
    "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8,
    "September": 9, "Oktober": 10, "November": 11, "Desember": 12
}

bulan_label = list(bulan_dict.keys())

tahun_input = st.selectbox("Tahun Prediksi", list(range(2015, 2031)), index=2025-2015)

# Input user
tahun = st.number_input("Tahun", value=datetime.now().year, min_value=2000, max_value=2100)
bulan = st.selectbox("Bulan", 
                     ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
                      "Juli", "Agustus", "September", "Oktober", "November", "Desember"], index=6)
BI_Rate = st.number_input("BI Rate (%)", value=6.0, step=0.01)
BBM = st.number_input("Harga BBM (Rp/L)", value=10000, step=50)
Kurs_USD_IDR = st.number_input("Kurs USD/IDR", value=15000, step=10)
Harga_Beras = st.number_input("Harga Beras (Rp/kg)", value=12000, step=50)
Inflasi_Inti = st.number_input("Inflasi Inti (%)", value=2.5, step=0.01)
Inflasi_Total = st.number_input("Inflasi Total (%)", value=2.7, step=0.01)

if st.button("Prediksi Inflasi"):
    input_user = {
        'Tahun': tahun,
        'Bulan': bulan,
        'BI_Rate': BI_Rate,
        'BBM': BBM,
        'Kurs_USD_IDR': Kurs_USD_IDR,
        'Harga_Beras': Harga_Beras,
        'Inflasi_Inti': Inflasi_Inti,
        'Inflasi_Total': Inflasi_Total
    }
    csv_path = 'data/data_inflasi.csv'
    model_path = 'model/model_inflasi.model'

    df_infer, df_histori = preprocess_and_update_histori(csv_path, input_user, features_training)
    prediksi = predict_inflasi(model_path, df_infer, features_training)

    # st.success(f"Prediksi Inflasi Total bulan depan: {prediksi:.2f}%")
    st.success(f"📌 Prediksi Inflasi untuk **{bulan} {tahun}** adalah: **{prediksi:.2f}%**")


