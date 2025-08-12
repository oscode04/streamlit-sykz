# dashboard_inferensi.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import Booster, DMatrix

from src.preprocessing import preprocess_and_update_histori
from src.inference import predict_inflasi

with open("data/features_training.txt") as f:
    features_training = [line.strip() for line in f.readlines()]

# Fungsi dari kode sebelumnya (preprocessing & update histori)
mapping_bulan = {
    'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
    'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
}

def encode_bulan(df, col='Bulan'):
    df['Bulan_Num'] = df[col].map(mapping_bulan)
    df['bulan_sin'] = np.sin(2 * np.pi * df['Bulan_Num'] / 12)
    df['bulan_cos'] = np.cos(2 * np.pi * df['Bulan_Num'] / 12)
    return df

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

def preprocess_and_update_histori(
    csv_path, input_user_dict,
    lag_columns=['BI_Rate', 'BBM', 'Kurs_USD_IDR', 'Harga_Beras', 'Inflasi_Inti', 'Inflasi_Total'],
    windows=[3,6,12], lags=[1,3,6,12]
):
    df_histori = pd.read_csv(csv_path)
    tahun = input_user_dict['Tahun']
    bulan = input_user_dict['Bulan']
    idx = df_histori[(df_histori['Tahun'] == tahun) & (df_histori['Bulan'] == bulan)].index
    if len(idx) > 0:
        df_histori.loc[idx[0], list(input_user_dict.keys())] = list(input_user_dict.values())
    else:
        df_histori = pd.concat([df_histori, pd.DataFrame([input_user_dict])], ignore_index=True)
    df_histori = encode_bulan(df_histori)
    df_histori = df_histori.sort_values(['Tahun', 'Bulan_Num']).reset_index(drop=True)
    df_histori = add_rolling_features(df_histori, lag_columns, windows)
    df_histori = generate_lag_features(df_histori, lag_columns, lags)
    df_histori = df_histori.dropna().reset_index(drop=True)
    df_infer = df_histori.iloc[[-1]]
    return df_infer, df_histori

def predict_inflasi(model_path, df_features):
    model = Booster()
    model.load_model(model_path)

    X = df_features.copy()

    # Drop kolom target dan kolom yang tidak numerik agar fitur hanya sesuai saat training
    drop_cols = ['Inflasi_Total', 'Bulan', 'Bulan_Num']
    for col in drop_cols:
        if col in X.columns:
            X = X.drop(columns=[col])

    dmatrix = DMatrix(X)
    preds = model.predict(dmatrix)
    return preds[0]


# ==============================
# Streamlit UI dan Logika
# ==============================
st.set_page_config(page_title="Prediksi Inflasi Bulanan", page_icon="📈", layout="wide")
st.title("📈 Dashboard Prediksi Inflasi Bulanan - XGBoost")
st.markdown("Masukkan nilai variabel makroekonomi untuk memprediksi inflasi bulan berikutnya.")

with st.sidebar:
    st.header("Input Parameter")
    tahun = st.number_input("Tahun", min_value=2000, max_value=2100, value=datetime.now().year)
    bulan = st.selectbox("Bulan", 
                         ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
                          "Juli", "Agustus", "September", "Oktober", "November", "Desember"], 
                         index=6)
    BI_Rate = st.number_input("BI Rate (%)", value=6.0, step=0.01)
    BBM = st.number_input("Harga BBM (Rp/L)", value=10000, step=50)
    Kurs_USD_IDR = st.number_input("Kurs USD/IDR", value=15000, step=10)
    Harga_Beras = st.number_input("Harga Beras (Rp/kg)", value=12000, step=50)
    Inflasi_Inti = st.number_input("Inflasi Inti (%)", value=2.5, step=0.01)
    Inflasi_Total = st.number_input("Inflasi Total (%)", value=2.7, step=0.01)  # input nilai Inflasi_Total terakhir juga

if st.sidebar.button("Prediksi Inflasi"):
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

    # Pastikan preprocess menerima feature_list agar langsung reorder
    df_infer, df_histori = preprocess_and_update_histori(csv_path, input_user, features_training)

    # Kalau belum reorder di fungsi, bisa juga reorder di sini (tapi harus pastikan semua kolom ada)
    # df_infer = reorder_features(df_infer, features_training)

    prediksi = predict_inflasi(model_path, df_infer)

    st.subheader("📊 Hasil Prediksi")
    st.metric(label="Prediksi Inflasi Bulanan (%)", value=f"{prediksi:.2f}")

    if prediksi > 5:
        st.warning("⚠️ Inflasi cukup tinggi, waspada kenaikan harga barang.")
    elif prediksi > 3:
        st.info("ℹ️ Inflasi dalam batas wajar namun perlu diwaspadai.")
    else:
        st.success("✅ Inflasi rendah dan terkendali.")

    with st.expander("Lihat Data Input"):
        st.dataframe(pd.DataFrame([input_user]))




