import streamlit as st
import pandas as pd
import requests
from datetime import datetime

PROVINCE_ID = 12  # Jawa Barat
# cat_1: Beras, cat_4: Telur Ayam Ras, cat_9: Minyak Goreng
COMCAT_ID = "cat_1,cat_4,cat_9" 

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bi_data(start_date: str, end_date: str):
    """
    Mengambil data harga komoditas dari API publik Bank Indonesia.
    """
    url = "https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataDaerah"
    params = {
        "price_type_id": 1,
        "comcat_id": COMCAT_ID,
        "province_id": PROVINCE_ID,
        "regency_id": "",
        "market_id": "",
        "tipe_laporan": 1,
        "start_date": start_date,
        "end_date": end_date,
        "_": str(int(datetime.now().timestamp() * 1000))
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        json_data = response.json()

        if 'data' not in json_data or not json_data['data']:
            st.warning("API Bank Indonesia tidak mengembalikan data untuk rentang tanggal yang dipilih.")
            return pd.DataFrame()
        
        return pd.DataFrame(json_data['data'])
    
    except requests.exceptions.RequestException as e:
        st.error(f"Gagal menghubungi API Bank Indonesia. Cek koneksi internet Anda. Detail: {e}")
        return pd.DataFrame()
    except ValueError:
        st.error("Gagal memproses respons dari API. Mungkin ada perubahan format dari sisi server.")
        return pd.DataFrame()

def reshape_and_clean_data(df_raw, commodity_details):
    """
    Mengubah data mentah (wide) menjadi data bersih (long) dan siap untuk feature engineering.
    Versi ini lebih robust dalam membedakan kolom tanggal dan memberikan pesan error yang jelas.
    """
    if df_raw.empty:
        return pd.DataFrame()

    # 1. Identifikasi kolom tanggal vs kolom identitas secara dinamis
    # Kolom tanggal adalah kolom yang mengandung karakter '/'
    date_columns = [col for col in df_raw.columns if isinstance(col, str) and '/' in col]
    # Kolom identitas adalah semua kolom sisanya
    id_vars = [col for col in df_raw.columns if col not in date_columns]
    
    if not date_columns:
        st.warning("Tidak ada kolom dengan format tanggal (dd/mm/yyyy) yang ditemukan dari data API.")
        return pd.DataFrame()

    # 2. Lakukan proses 'melt' untuk mengubah format tabel dari lebar ke panjang
    df_long = df_raw.melt(
        id_vars=id_vars, 
        value_vars=date_columns,
        var_name='date_str', 
        value_name='harga'
    ).rename(columns={'name': 'komoditas_sub'}) # Ganti nama kolom 'name' agar konsisten
    
    # Simpan nama asli dari API sebelum difilter, untuk keperluan debugging
    original_names_from_api = df_long['komoditas_sub'].unique()

    # 3. Filter data hanya untuk sub-komoditas target yang ada di config
    df_long_filtered = df_long[df_long['komoditas_sub'].isin(commodity_details['targets'])]

    # 4. Validasi krusial: Cek apakah data kosong setelah difilter
    if df_long_filtered.empty:
        st.error(f"Tidak ada data yang cocok setelah filter untuk kelompok '{commodity_details['main']}'.", icon="🚨")
        st.info("Ini kemungkinan besar karena nama sub-komoditas di `src/config.py` tidak cocok persis dengan nama dari API.")
        
        # Tampilkan perbandingan untuk mempermudah debugging
        st.markdown("**Nama di `config.py` (yang dicari):**")
        st.code(f"{commodity_details['targets']}", language='python')
        st.markdown("**Nama yang Tersedia dari API:**")
        st.code(f"{list(original_names_from_api)}", language='python')
        
        return pd.DataFrame() # Kembalikan DataFrame kosong agar proses di app.py berhenti

    # 5. Proses pembersihan data jika data ditemukan
    df_final = df_long_filtered.copy() 
    
    # Bersihkan kolom harga dari koma dan konversi ke numerik
    df_final['harga'] = df_final['harga'].astype(str).str.replace(',', '', regex=False)
    df_final['harga'] = pd.to_numeric(df_final['harga'], errors='coerce')
    
    # Konversi kolom tanggal string ke format datetime
    df_final['date'] = pd.to_datetime(df_final['date_str'], format='%d/%m/%Y', errors='coerce')
    
    # Buang baris yang gagal dikonversi dan urutkan berdasarkan tanggal
    df_final = df_final.sort_values('date').dropna(subset=['harga', 'date'])
    
    # 6. Kembalikan DataFrame bersih dengan kolom yang relevan
    return df_final[['date', 'komoditas_sub', 'harga']]