import streamlit as st
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBRegressor, Booster, DMatrix
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ======================
# 1. Load Model
# ======================
MODEL_PATH = "model_inflasi.pkl"  # atau "final_model.pkl"

@st.cache_resource
def load_model():
    if MODEL_PATH.endswith(".json"):
        model = Booster()
        model.load_model(MODEL_PATH)
        return model
    else:
        return joblib.load(MODEL_PATH)

model = load_model()

# ======================
# 2. Sidebar Menu
# ======================
st.sidebar.title("📊 Menu")
menu = st.sidebar.radio("Pilih Halaman:", ["Prediksi Manual", "Prediksi dari File", "Evaluasi Model"])

# ======================
# 3. Halaman Prediksi Manual
# ======================
if menu == "Prediksi Manual":
    st.title("🔮 Prediksi Inflasi (Input Manual)")

    # List fitur
    features = ['BI_Rate', 'BBM', 'Kurs_USD_IDR', 'Harga_Beras', 'Inflasi_Inti', 'Inflasi_Total']
    input_data = {}

    for feat in features:
        input_data[feat] = st.number_input(f"{feat}", value=0.0)

    if st.button("Prediksi"):
        df_input = pd.DataFrame([input_data])

        if isinstance(model, Booster):
            dmatrix_input = DMatrix(df_input, feature_names=df_input.columns)
            pred = model.predict(dmatrix_input)[0]
        else:
            pred = model.predict(df_input)[0]

        st.success(f"📈 Hasil Prediksi: {pred:.2f}")

# ======================
# 4. Halaman Prediksi dari File
# ======================
elif menu == "Prediksi dari File":
    st.title("📂 Prediksi Inflasi (Upload CSV)")

    uploaded_file = st.file_uploader("Upload file CSV dengan kolom fitur lengkap", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data yang diupload:")
        st.dataframe(df.head())

        if isinstance(model, Booster):
            dmatrix_input = DMatrix(df, feature_names=df.columns)
            preds = model.predict(dmatrix_input)
        else:
            preds = model.predict(df)

        df['Prediksi'] = preds
        st.write("📊 Hasil Prediksi:")
        st.dataframe(df)
        st.download_button("Download Hasil", df.to_csv(index=False), "prediksi.csv", "text/csv")

# ======================
# 5. Halaman Evaluasi Model
# ======================
elif menu == "Evaluasi Model":
    st.title("📏 Evaluasi Model XGBoost")

    try:
        X_test, y_test = joblib.load("test_data.pkl")

        if isinstance(model, Booster):
            dmatrix_test = DMatrix(X_test, label=y_test, feature_names=X_test.columns)
            y_pred = model.predict(dmatrix_test)
        else:
            y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        st.metric("MAE", f"{mae:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")
        st.metric("R²", f"{r2:.4f}")
        st.metric("MAPE", f"{mape:.2f}%")
    except FileNotFoundError:
        st.error("❌ File test_data.pkl tidak ditemukan. Upload dulu file tersebut.")
