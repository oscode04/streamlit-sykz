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

    # Pastikan ada cukup data historis
    if df_histori.shape[0] < max(lags) + 1:
        raise ValueError(f"Data historis harus minimal {max(lags)+1} baris supaya lag bisa terbentuk.")

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

    # Jangan dropna, tapi isi nan dengan 0 agar lag dan rolling tetap ada
    df_histori = df_histori.fillna(0).reset_index(drop=True)

    # Debug print kolom yang ada setelah feature engineering
    print("Kolom-kolom setelah fitur dibuat:")
    print(df_histori.columns.tolist())

    df_infer = df_histori.iloc[[-1]]
    df_infer = reorder_features(df_infer, feature_list)

    return df_infer, df_histori
