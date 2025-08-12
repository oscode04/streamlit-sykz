from xgboost import Booster, DMatrix

def predict_inflasi(model_path, df_features, feature_list):
    """
    model_path: path ke file model .model hasil save XGBoost
    df_features: DataFrame 1 baris fitur lengkap hasil preprocessing siap inferensi
    feature_list: list nama kolom fitur lengkap yang dipakai saat training (urutan harus sama)
    """
    model = Booster()
    model.load_model(model_path)
    
    # Pastikan kolom fitur lengkap dan urut
    X = df_features.copy()
    X = X[feature_list]
    
    dmatrix = DMatrix(X)
    preds = model.predict(dmatrix)
    return preds[0]
