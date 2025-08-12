from xgboost import Booster, DMatrix

def predict_inflasi(model_path, df_features, model_feature_cols):
    """
    model_path: path ke file model .model hasil save XGBoost
    df_features: DataFrame 1 baris fitur lengkap hasil preprocessing siap inferensi
    model_feature_cols: list nama kolom fitur lengkap yang dipakai saat training (urutan harus sama)
    """
    model = Booster()
    model.load_model(model_path)
    
    # Ambil hanya fitur yang diperlukan sesuai urutan training
    X = df_features.copy()
    X = X[model_feature_cols]
    
    dmatrix = DMatrix(X)
    preds = model.predict(dmatrix)
    return preds[0]
