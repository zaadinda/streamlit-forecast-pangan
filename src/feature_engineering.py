import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype

LAGS = [1, 2, 3, 7, 30]
WINDOWS = [7, 30]
SEQ_LENGTH = 30

def create_date_features(start_date, end_date):
    """
    Membuat fitur berbasis tanggal (termasuk one-hot encoding).
    """
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df_date_features = pd.DataFrame(index=all_dates)

    df_date_features['day_of_week'] = df_date_features.index.dayofweek
    df_date_features['month'] = df_date_features.index.month
    df_date_features['is_weekend'] = df_date_features['day_of_week'].isin([5, 6]).astype(int)
    df_date_features['trend'] = np.arange(len(df_date_features))

    # 1. Definisikan semua kategori yang mungkin ada untuk hari dan bulan
    day_of_week_type = CategoricalDtype(categories=range(7), ordered=True)
    month_type = CategoricalDtype(categories=range(1, 13), ordered=True)
    
    # 2. Terapkan tipe data kategorikal tersebut ke kolom
    df_date_features['day_of_week'] = df_date_features['day_of_week'].astype(day_of_week_type)
    df_date_features['month'] = df_date_features['month'].astype(month_type)
    
    # 3. One-Hot Encoding
    # get_dummies akan selalu membuat 7 kolom untuk hari dan 12 kolom untuk bulan, mengisi dengan 0 jika kategori tersebut tidak ada dalam data.
    df_date_features = pd.get_dummies(df_date_features, columns=['day_of_week', 'month'], drop_first=False)
    
    return df_date_features


def add_lag_and_rolling_features(df_pivot, target_cols):
    """
    Menambahkan fitur lag dan rolling window ke data harga yang sudah di-pivot.
    """
    feature_dfs = []
    for col in target_cols:
        # Lag features
        for lag in LAGS:
            feature_dfs.append(df_pivot[col].shift(lag).rename(f'{col}_lag_{lag}'))
        # Rolling window features
        for window in WINDOWS:
            feature_dfs.append(df_pivot[col].rolling(window=window).mean().rename(f'{col}_rolling_mean_{window}'))
            
    return pd.concat(feature_dfs, axis=1)

def full_preparation_pipeline(df_long, commodity_details):
    """
    Menjalankan seluruh pipeline persiapan data: pivot, feature engineering, dan pembersihan.
    Mengembalikan sequence terakhir yang siap untuk prediksi.
    """
    target_cols = commodity_details['targets']
    
    # 1. Pivot data
    df_pivot = df_long.pivot_table(index='date', columns='komoditas_sub', values='harga')
    
    # Pastikan semua kolom target ada, isi dengan NaN jika tidak ada lalu interpolasi
    for col in target_cols:
        if col not in df_pivot:
            df_pivot[col] = np.nan
    
    df_pivot = df_pivot.asfreq('D')
    df_pivot[target_cols] = df_pivot[target_cols].interpolate(method='linear', limit_direction='both')

    # 2. Buat fitur tanggal
    df_date_features = create_date_features(df_pivot.index.min(), df_pivot.index.max())

    # 3. Tambahkan fitur lag dan rolling
    df_lag_roll_features = add_lag_and_rolling_features(df_pivot, target_cols)

    # 4. Gabungkan semua data: harga (target) + fitur
    df_combined = pd.concat([df_pivot[target_cols], df_date_features, df_lag_roll_features], axis=1)

    # 5. Drop baris dengan NaN yang dihasilkan oleh lag/rolling
    df_processed = df_combined.dropna()
    
    if len(df_processed) < SEQ_LENGTH:
        # returns 3 values: (None untuk sequence, None untuk feature_cols, dan pesan error)
        error_message = f"Data historis tidak cukup untuk prediksi. Dibutuhkan {SEQ_LENGTH} hari data valid setelah feature engineering, hanya tersedia {len(df_processed)} hari."
        return None, None, error_message
 
    # 6. Ambil sekuens terakhir dan daftar kolom fitur
    prediction_sequence = df_processed.tail(SEQ_LENGTH)
    feature_cols = [col for col in df_processed.columns if col not in target_cols]
    
    # Pastikan urutan kolom sesuai dengan saat training
    prediction_sequence = prediction_sequence[target_cols + feature_cols]
    
    # Jika berhasil, kembalikan 3 nilai (sequence, feature_cols, dan None untuk pesan error)
    return prediction_sequence, feature_cols, None