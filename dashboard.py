import streamlit as st
import pandas as pd
import numpy as np
import json
import xgboost as xgb

# ==============================
# 1. Load model & fitur
# ==============================
model = xgb.Booster()
model.load_model("model_inflasi.json")  # hasil dari final_model.save_model()

with open("feature_columns.json", "r") as f:
    feature_cols = json.load(f)

# Fitur yang perlu rolling/lag
lag_columns = ['BI_Rate', 'BBM', 'Kurs_USD_IDR', 'Harga_Beras',
               'Inflasi_Inti', 'Inflasi_Total', 'bulan_sin', 'bulan_cos']

# ==============================
# 2. Fungsi feature engineering
# ==============================
def add_rolling_features(df, columns, windows=[3, 6, 12]):
    for col in columns:
        for win in windows:
            df[f'{col}_roll_mean_{win}'] = df[col].rolling(window=win).mean()
            df[f'{col}_roll_std_{win}'] = df[col].rolling(window=win).std()
    return df

def generate_lag_features(df, columns, lags=[1, 3, 6, 12]):
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

# ==============================
# 3. Judul dashboard
# ==============================
st.title("Prediksi Inflasi - Dashboard")

# ==============================
# 4. Upload data historis
# ==============================
st.subheader("Upload Data Historis")
uploaded_file = st.file_uploader("Upload file CSV historis", type="csv")

if uploaded_file:
    df_hist = pd.read_csv(uploaded_file)

    # Pastikan kolom wajib ada
    required_cols = ["Tahun", "BI_Rate", "BBM", "Kurs_USD_IDR", 
                     "Harga_Beras", "Inflasi_Inti", "Inflasi_Total", 
                     "bulan_sin", "bulan_cos"]
    if not all(col in df_hist.columns for col in required_cols):
        st.error(f"CSV harus punya kolom: {required_cols}")
    else:
        st.success("Data historis berhasil dibaca!")

        # ==============================
        # 5. Input data terbaru
        # ==============================
        st.subheader("Masukkan Data Bulan Terbaru")
        tahun = st.number_input("Tahun", min_value=2000, max_value=2100, value=2025)
        bulan = st.slider("Bulan", 1, 12, 1)
        bi_rate = st.number_input("BI Rate", value=5.75)
        bbm = st.number_input("Harga BBM (Rp/liter)", value=10000)
        kurs = st.number_input("Kurs USD/IDR", value=15000)
        harga_beras = st.number_input("Harga Beras (Rp/kg)", value=14000)
        inflasi_inti = st.number_input("Inflasi Inti (%)", value=3.0)
        inflasi_total = st.number_input("Inflasi Total (%)", value=4.0)

        bulan_sin = np.sin(2 * np.pi * bulan / 12)
        bulan_cos = np.cos(2 * np.pi * bulan / 12)

        # Buat dataframe 1 baris untuk prediksi
        new_row = pd.DataFrame([[
            tahun, bi_rate, bbm, kurs, harga_beras, 
            inflasi_inti, inflasi_total, bulan_sin, bulan_cos
        ]], columns=required_cols)

        # Gabungkan dengan historis
        df_full = pd.concat([df_hist, new_row], ignore_index=True)

        # ==============================
        # 6. Feature engineering sama persis
        # ==============================
        df_full = add_rolling_features(df_full, lag_columns)
        df_full = generate_lag_features(df_full, lag_columns)
        df_full = df_full.dropna().reset_index(drop=True)

        # Ambil baris terakhir untuk prediksi
        X_pred = df_full[feature_cols].iloc[[-1]]

        # ==============================
        # 7. Prediksi
        # ==============================
        if st.button("Prediksi"):
            dmatrix_input = xgb.DMatrix(X_pred, feature_names=list(feature_cols))
            prediction = model.predict(dmatrix_input)[0]

            st.subheader("Hasil Prediksi Inflasi:")
            st.metric(label="Prediksi (%)", value=f"{prediction:.2f}")

            st.subheader("Data Fitur yang Digunakan:")
            st.dataframe(X_pred)
