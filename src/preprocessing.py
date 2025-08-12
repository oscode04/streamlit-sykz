import pandas as pd
import numpy as np

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
    lag_columns=['BI_Rate', 'BBM', 'Kurs_USD_IDR', 'Harga_Beras', 'Inflasi_Inti', 'Inflasi_Total', 'bulan_sin', 'bulan_cos'],
    windows=[3, 6, 12], lags=[1, 3, 6, 12],
    model_feature_cols=None  # tambahan parameter list kolom fitur yang diharapkan model
):
    df_histori = pd.read_csv(csv_path)
    tahun = input_user_dict['Tahun']
    bulan = input_user_dict['Bulan']

    # Update atau tambah data baru
    idx = df_histori[(df_histori['Tahun'] == tahun) & (df_histori['Bulan'] == bulan)].index
    if len(idx) > 0:
        df_histori.loc[idx[0], list(input_user_dict.keys())] = list(input_user_dict.values())
    else:
        df_histori = pd.concat([df_histori, pd.DataFrame([input_user_dict])], ignore_index=True)

    # Encode bulan
    df_histori = encode_bulan(df_histori)

    # Sort
    df_histori = df_histori.sort_values(['Tahun', 'Bulan_Num']).reset_index(drop=True)

    # Tambah fitur roll dan lag
    df_histori = add_rolling_features(df_histori, lag_columns, windows)
    df_histori = generate_lag_features(df_histori, lag_columns, lags)

    # Drop baris NaN (karena rolling dan lag menghasilkan NaN di awal)
    df_histori = df_histori.dropna().reset_index(drop=True)

    # Ambil baris terakhir untuk inferensi
    df_infer = df_histori.iloc[[-1]].copy()

    # Hapus kolom yang tidak dipakai model: target dan kolom Bulan, Bulan_Num
    drop_cols = ['Inflasi_Total', 'Bulan', 'Bulan_Num']
    for col in drop_cols:
        if col in df_infer.columns:
            df_infer = df_infer.drop(columns=[col])

    # **KOREKSI PENTING: pastikan urutan dan kolom fitur sesuai dengan yang dipakai saat training**
    if model_feature_cols is not None:
        # Buat df_infer hanya dengan kolom yang dipakai model (urut sesuai list)
        missing_cols = set(model_feature_cols) - set(df_infer.columns)
        if missing_cols:
            raise ValueError(f"Fitur hilang di data inferensi: {missing_cols}")

        df_infer = df_infer[model_feature_cols]

    return df_infer, df_histori
