import streamlit as st
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

from .feature_engineering import create_date_features, add_lag_and_rolling_features

@st.cache_resource(show_spinner=False)
def load_all_models_and_scalers(config):
    models = {}
    scalers = {}
    for commodity, details in config.items():
        try:
            models[commodity] = load_model(details['model_path'])
            scalers[commodity] = {
                'X': joblib.load(details['scaler_x_path']),
                'y': joblib.load(details['scaler_y_path'])
            }
        except FileNotFoundError as e:
            st.error(f"Gagal memuat file untuk {commodity}: {e}.")
            return None, None
    return models, scalers

def forecast_iteratively(model, scalers, initial_sequence, feature_cols, target_cols, future_steps=30):
    """
    Melakukan forecasting iteratif untuk beberapa hari ke depan (misal: 30 hari).
    """
    scaler_X = scalers['X']
    scaler_y = scalers['y']
    
    current_sequence_df = initial_sequence.copy()
    all_predictions = []

    for i in range(future_steps):
        # 1. Siapkan input untuk prediksi saat ini
        features_to_scale = current_sequence_df[feature_cols].values
        scaled_features = scaler_X.transform(features_to_scale)
        model_input = scaled_features.reshape(1, len(current_sequence_df), len(feature_cols))

        # 2. Prediksi 1 langkah ke depan
        predicted_scaled = model.predict(model_input)
        predicted_prices = scaler_y.inverse_transform(predicted_scaled)
        all_predictions.append(predicted_prices[0])

        # --- 3. Siapkan sekuens untuk iterasi berikutnya ---
        # Tentukan tanggal untuk prediksi ini
        next_date = current_sequence_df.index[-1] + pd.Timedelta(days=1)
        
        # Buat baris baru dengan harga yang baru saja diprediksi
        new_row_prices = pd.DataFrame(predicted_prices, index=[next_date], columns=target_cols)
        
        # Gabungkan dengan histori harga untuk re-kalkulasi fitur
        price_history_updated = pd.concat([current_sequence_df[target_cols], new_row_prices])
        
        # Buat ulang fitur tanggal dan lag/rolling untuk baris baru ini
        date_features = create_date_features(next_date, next_date)
        lag_roll_features = add_lag_and_rolling_features(price_history_updated, target_cols).tail(1)
        
        # Gabungkan semua menjadi baris fitur yang lengkap
        new_full_row = pd.concat([new_row_prices, date_features, lag_roll_features], axis=1)
        
        # Tambahkan baris baru ke sekuens dan hapus baris terlama
        current_sequence_df = pd.concat([current_sequence_df, new_full_row])
        current_sequence_df = current_sequence_df.iloc[1:]

    return np.array(all_predictions)