import pandas as pd
import numpy as np

# Mapping nama bulan ke angka
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
    # 1. Load histori CSV
    df_histori = pd.read_csv(csv_path)

    # 2. Update atau tambah data input user
    tahun = input_user_dict['Tahun']
    bulan = input_user_dict['Bulan']
    # Cek apakah data sudah ada
    idx = df_histori[(df_histori['Tahun'] == tahun) & (df_histori['Bulan'] == bulan)].index
    if len(idx) > 0:
        # Update baris existing
        df_histori.loc[idx[0], list(input_user_dict.keys())] = list(input_user_dict.values())
    else:
        # Tambah baris baru
        df_histori = pd.concat([df_histori, pd.DataFrame([input_user_dict])], ignore_index=True)

    # 3. Sort data berdasarkan Tahun dan Bulan (agar rolling/lag benar)
    # Ubah bulan ke angka dulu
    df_histori = encode_bulan(df_histori)
    df_histori = df_histori.sort_values(['Tahun', 'Bulan_Num']).reset_index(drop=True)

    # 4. Tambah fitur rolling dan lag
    df_histori = add_rolling_features(df_histori, lag_columns, windows)
    df_histori = generate_lag_features(df_histori, lag_columns, lags)

    # 5. Drop baris dengan NaN (rolling/lag di awal data)
    df_histori = df_histori.dropna().reset_index(drop=True)

    # 6. Ambil baris terakhir (input user sudah termasuk) sebagai data siap inferensi
    df_infer = df_histori.iloc[[-1]]

    return df_infer, df_histori
