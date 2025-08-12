import streamlit as st
from src.preprocessing import preprocess_and_update_histori
from src.inference import predict_inflasi
from datetime import datetime

# Load fitur training yang sudah lengkap dan urut
with open('data/features_training.txt') as f:
    features_training = [line.strip() for line in f.readlines()]

st.title("📈 Prediksi Inflasi - Dashboard Forecasting")

# ====== CSS untuk kotak input ======
st.markdown("""
<style>
.input-box {
    border: 2px solid white;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    margin-bottom: 20px;
}
.input-box h4 {
    margin-top: 0;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Mulai kotak input
st.markdown('<div class="input-box">', unsafe_allow_html=True)
st.markdown("#### 📦 Input Data Ekonomi (Bulan Sebelumnya)", unsafe_allow_html=True)

# Semua input berada di dalam kotak
tahun = st.number_input("Tahun", value=datetime.now().year, min_value=2010, max_value=2030)
bulan = st.selectbox(
    "Bulan", 
    ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
     "Juli", "Agustus", "September", "Oktober", "November", "Desember"], 
    index=6
)
BI_Rate = st.number_input("BI Rate (%)", value=6.0, step=0.01)
BBM = st.number_input("Harga BBM (Rp/L)", value=10000, step=50)
Kurs_USD_IDR = st.number_input("Kurs USD/IDR", value=15000, step=10)
Harga_Beras = st.number_input("Harga Beras (Rp/kg)", value=12000, step=50)
Inflasi_Inti = st.number_input("Inflasi Inti (%)", value=2.5, step=0.01)
Inflasi_Total = st.number_input("Inflasi Total (%)", value=2.7, step=0.01)

# Tombol prediksi di dalam kotak
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
    
    df_infer, df_histori = preprocess_and_update_histori(
        'data/data_inflasi.csv', input_user, features_training
    )
    prediksi = predict_inflasi(
        'model/model_inflasi.model', df_infer, features_training
    )
    
    st.success(f"📌 Prediksi Inflasi untuk **{bulan} {tahun}** adalah: **{prediksi:.2f}%**")

# Tutup kotak input
st.markdown("</div>", unsafe_allow_html=True)
