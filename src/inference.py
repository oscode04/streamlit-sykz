from xgboost import Booster, DMatrix

def predict_inflasi(model_path, df_features):
    """
    model_path: path ke file model .model hasil save XGBoost
    df_features: DataFrame 1 baris fitur lengkap hasil preprocessing siap inferensi
    """
    # Load model
    model = Booster()
    model.load_model(model_path)
    
    # Copy dataframe supaya tidak mengubah aslinya
    X = df_features.copy()

    # Drop kolom target dan kolom 'Bulan' yang bertipe object, supaya hanya fitur numerik yang masuk
    cols_to_drop = []
    if 'Inflasi_Total' in X.columns:
        cols_to_drop.append('Inflasi_Total')
    if 'Bulan' in X.columns:
        cols_to_drop.append('Bulan')

    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
    
    # Buat DMatrix dari fitur
    dmatrix = DMatrix(X)
    
    # Prediksi
    preds = model.predict(dmatrix)
    
    # Ambil prediksi pertama (karena hanya 1 baris input)
    return preds[0]
