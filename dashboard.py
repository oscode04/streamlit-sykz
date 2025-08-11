import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
# Load model
model = joblib.load("model_inflasi_percobaan.pkl")

# ===== Header App =====
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

bulan_input = st.selectbox("Bulan Prediksi", bulan_label, index=7)
tahun_input = st.selectbox("Tahun Prediksi", list(range(2015, 2031)), index=2025-2015)

st.markdown(f"📝 Masukkan data ekonomi untuk **bulan sebelumnya** dari {bulan_input} {tahun_input}")
# ===== Form Input =====
with st.form("input_form"):
    st.subheader("📊 Input Data Ekonomi (Bulan Sebelumnya)")

    bi_rate = st.number_input("BI Rate (bulan sebelumnya)", value=5.25, format="%.2f")
    bbm = st.number_input("Harga BBM (bulan sebelumnya)", value=10000.0, format="%.2f")

    kurs = st.number_input("Kurs USD/IDR (bulan sebelumnya)", value=16286.0, format="%.2f")
    beras = st.number_input("Harga Beras/kg (bulan sebelumnya)", value=14202.0, format="%.2f")
    inflasi_inti = st.number_input("Inflasi Inti (bulan sebelumnya)", value=0.13, format="%.2f")
    inflasi_total = st.number_input("Inflasi Total (bulan sebelumnya)", value=2.37, format="%.2f")

    submitted = st.form_submit_button("🎯 Prediksi")

# ===== Prediksi =====
if submitted:
    # Buat DataFrame input
    input_data = pd.DataFrame([{
        'BI_Rate_lag1': bi_rate,
        'BBM_lag1': bbm,
        'Kurs_USD_IDR_lag1': kurs,
        'Harga_Beras_lag1': beras,
        'Inflasi_Inti_lag1': inflasi_inti,
        'Inflasi_Total_lag1': inflasi_total
    }])
    # Prediksi
    prediction = model.predict(input_data)[0]

    st.success(f"📌 Prediksi Inflasi Total untuk **{bulan_input} {tahun_input}** adalah: **{prediction:.2f}%**")