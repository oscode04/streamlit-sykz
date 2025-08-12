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

def reorder_features(df, feature_list):
    # Tambahkan kolom kosong jika belum ada, lalu urutkan kolom sesuai feature_list
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_list]
    return df

def preprocess_and_update_histori(
    csv_path, input_user_dict, feature_list,
    lag_columns=['BI_Rate', 'BBM', 'Kurs_USD_IDR', 'Harga_Beras', 'Inflasi_Inti', 'Inflasi_Total', 'bulan_sin', 'bulan_cos'],
    windows=[3,6,12], lags=[1,3,6,12]
):
    df_histori = pd.read_csv(csv_path)

    tahun = input_user_dict['Tahun']
    bulan = input_user_dict['Bulan']

    # Update atau tambah data input_user
    idx = df_histori[(df_histori['Tahun'] == tahun) & (df_histori['Bulan'] == bulan)].index
    if len(idx) > 0:
        df_histori.loc[idx[0], list(input_user_dict.keys())] = list(input_user_dict.values())
    else:
        df_histori = pd.concat([df_histori, pd.DataFrame([input_user_dict])], ignore_index=True)

    # Encode bulan dulu
    df_histori = encode_bulan(df_histori)

    # Urutkan berdasarkan Tahun dan Bulan_Num
    df_histori = df_histori.sort_values(['Tahun', 'Bulan_Num']).reset_index(drop=True)

    # 1. Generate lag features dulu
    df_histori = generate_lag_features(df_histori, lag_columns, lags)

    # 2. Tentukan kolom untuk rolling: lag_columns plus lag columns
    rolling_cols = lag_columns.copy()
    for col in lag_columns:
        for lag in lags:
            col_lag = f"{col}_lag{lag}"
            if col_lag in df_histori.columns:
                rolling_cols.append(col_lag)

    # 3. Generate rolling features (mean, std)
    df_histori = add_rolling_features(df_histori, rolling_cols, windows)

    # 4. Isi NaN dengan 0 agar kolom lag dan rolling tetap lengkap dan tidak error
    df_histori = df_histori.fillna(0)

    # 5. Ambil baris terakhir untuk inferensi
    df_infer = df_histori.iloc[[-1]]

    # 6. Reorder fitur sesuai feature_list (tambahkan kolom kosong jika tidak ada)
    for col in feature_list:
        if col not in df_infer.columns:
            df_infer[col] = 0
    df_infer = df_infer[feature_list]

    return df_infer, df_histori

