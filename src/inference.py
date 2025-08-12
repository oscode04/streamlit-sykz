from xgboost import Booster, DMatrix

def predict_inflasi(model_path, df_features):
    """
    model_path: path ke file model .model hasil save XGBoost
    df_features: DataFrame 1 baris fitur lengkap hasil preprocessing siap inferensi
    """
    # Load model
    model = Booster()
    model.load_model(model_path)
    
    # Buat DMatrix dari fitur (hapus kolom target jika ada)
    # Asumsi kolom target 'Inflasi_Total' juga ada di df_features, kita drop supaya hanya fitur saja
    if 'Inflasi_Total' in df_features.columns:
        X = df_features.drop(columns=['Inflasi_Total'])
    else:
        X = df_features.copy()
    
    dmatrix = DMatrix(X)
    
    # Prediksi
    preds = model.predict(dmatrix)
    
    # Prediksi hanya 1 baris, jadi ambil elemen pertama
    return preds[0]
