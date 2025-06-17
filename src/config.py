COMMODITY_CONFIG = {
    "Beras": {
        "main": "Beras",
        "targets": [
            "Beras Kualitas Bawah I", "Beras Kualitas Bawah II", 
            "Beras Kualitas Medium I", "Beras Kualitas Medium II", 
            "Beras Kualitas Super I", "Beras Kualitas Super II"
        ],
        # Model terbaik untuk Beras adalah 'stacked'
        "model_path": "models/Beras_stacked_model.h5",
        "scaler_x_path": "models/Beras_scaler_X.pkl",
        "scaler_y_path": "models/Beras_scaler_y.pkl"
    },
    "Telur Ayam": {
        "main": "Telur Ayam",
        "targets": ["Telur Ayam Ras Segar"],
        # Model terbaik untuk Telur Ayam adalah 'bidirectional'
        "model_path": "models/Telur Ayam_bidirectional_model.h5",
        "scaler_x_path": "models/Telur Ayam_scaler_X.pkl",
        "scaler_y_path": "models/Telur Ayam_scaler_y.pkl"
    },
    "Minyak Goreng": {
        "main": "Minyak Goreng",
        "targets": [
            "Minyak Goreng Curah", 
            "Minyak Goreng Kemasan Bermerk 1", 
            "Minyak Goreng Kemasan Bermerk 2"
        ],
        # Model terbaik untuk Minyak Goreng adalah 'baseline'
        "model_path": "models/Minyak Goreng_baseline_model.h5",
        "scaler_x_path": "models/Minyak Goreng_scaler_X.pkl",
        "scaler_y_path": "models/Minyak Goreng_scaler_y.pkl"
    }
}