import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go


# -----------------------------------------------------------------------------
# 1. IMPOR & KONFIGURASI AWAL
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Proyeksi Harga Pangan",
    page_icon="📈"
)


try:
    from src.config import COMMODITY_CONFIG
    from src.data_handler import fetch_bi_data, reshape_and_clean_data
    from src.feature_engineering import full_preparation_pipeline
    from src.predictions import load_all_models_and_scalers, forecast_iteratively
except ImportError as e:
    st.error(f"Gagal mengimpor modul dari folder 'src'. Pastikan struktur folder sudah benar. Detail: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# 2. LOAD MODEL 
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_dependencies():
    """Memuat model dan scaler sekali saja."""
    try:
        models, scalers = load_all_models_and_scalers(COMMODITY_CONFIG)
        return models, scalers
    except (FileNotFoundError, TypeError) as e:
        st.error(f"Gagal memuat file model/scaler. Pastikan file ada di 'models/' dan path di 'src/config.py' benar. Detail: {e}")
        st.stop()


models, scalers = load_models_and_dependencies()

# =============================================================================
# 3. HELPER UNTUK UI
# =============================================================================
def display_results(results):
    """
    Fungsi untuk menampilkan semua hasil prediksi dalam layout tab yang rapi.
    """
    st.success("✅ Proyeksi berhasil dibuat!")
    st.markdown("---")
   
    df_forecast = results['df_forecast']
    sequence = results['sequence']
    details = results['details']
   
    tab1, tab2, tab3 = st.tabs(["📊 Ringkasan Proyeksi", "📈 Grafik Tren", "📋 Data Detail"])

    with tab1:
        st.subheader(f"Highlights Proyeksi untuk {details['main']}")

        besok = datetime.now() + timedelta(days=1)
        hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
        bulan = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
                 "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
        tanggal_lengkap = f"{hari[besok.weekday()]}, {besok.day} {bulan[besok.month]} {besok.year}"
        st.markdown(f"Berikut adalah estimasi harga untuk esok hari: **{tanggal_lengkap}**.")

        st.markdown("""
            <style>
            .metric-grid-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 1.5rem;
                padding: 1rem 0;
            }
            div.stMetric {
                border: 1px solid rgba(0,0,0,0.1);
                border-radius: 10px;
                padding: 1.5rem;
                background-color: #fafafa;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                transition: all 0.2s ease-in-out;
            }
            div.stMetric:hover {
                transform: scale(1.03);
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            }
            div.stMetric > div {
                justify-content: center;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="metric-grid-container">', unsafe_allow_html=True)
        next_day_prices = df_forecast.iloc[0]
        for target_name in details['targets']:
            st.metric(
                label=target_name,
                value=f"Rp {next_day_prices[target_name]:,.0f}"
            )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f"**Proyeksi ini adalah hasil estimasi model matematis, bukan merupakan jaminan harga di masa depan.*")


    with tab2:
        st.subheader("Grafik Tren Harga Historis vs. Harga Proyeksi 30 Hari")
       
        history_to_plot = sequence[details['targets']]
        fig = go.Figure()
       
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


        for i, col in enumerate(details['targets']):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=history_to_plot.index, y=history_to_plot[col],
                mode='lines', name=f'Historis - {col}',
                line=dict(color=color)
            ))
            fig.add_trace(go.Scatter(
                x=df_forecast.index, y=df_forecast[col],
                mode='lines', name=f'Proyeksi - {col}',
                line=dict(dash='dash', color=color)
            ))
       
        fig.add_vline(x=history_to_plot.index[-1].value, line_width=2, line_dash="dot", line_color="grey",
                      annotation_text="Mulai Proyeksi", annotation_position="top right")

        fig.update_layout(
            
            # setting margin
            margin=dict(t=120, b=80), # t = top, b = bottom

            xaxis_title='Tanggal',
            yaxis_title='Harga (Rp)',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02, 
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)


    with tab3:
        st.subheader("Tabel Detail Proyeksi Harga 30 Hari")
        st.dataframe(df_forecast.style.format("Rp {:,.2f}"), use_container_width=True)
       
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=True).encode('utf-8')


        csv = convert_df_to_csv(df_forecast)
        st.download_button(
           label="📥 Download Data Proyeksi (CSV)",
           data=csv,
           file_name=f"proyeksi_{details['main']}_{datetime.now().strftime('%Y%m%d')}.csv",
           mime="text/csv",
        )


# =============================================================================
# 4. MAIN UI
# =============================================================================
if 'prediction_generated' not in st.session_state:
    st.session_state.prediction_generated = False
    st.session_state.results = None

with st.sidebar:
    st.title("⚙️ Parameter Proyeksi")
    st.markdown("Tentukan parameter untuk menghasilkan proyeksi harga.")
   
    selected_commodity = st.selectbox(
        "Pilih Kelompok Komoditas",
        list(COMMODITY_CONFIG.keys()),
        key="commodity_choice"
    )
    details = COMMODITY_CONFIG[selected_commodity]
   
    with st.expander("Lihat Detail Sub-Komoditas"):
        for target in details['targets']:
            st.markdown(f"- {target}")


    st.subheader("🗓️ Periode Analisis")
    st.info("Pilih rentang waktu data historis sebagai dasar analisis. Minimal 60 hari untuk hasil optimal.", icon="💡")
   
    today = datetime.now()
    default_start_date = today - timedelta(days=90)
   
    start_date = st.date_input("Dari Tanggal", value=default_start_date)
    end_date = st.date_input("Hingga Tanggal", value=today)
   
    st.markdown("---")
    if st.button("🏷️ Cek Proyeksi Harga", type="primary", use_container_width=True):
        if start_date > end_date:
            st.error("Tanggal mulai tidak boleh melebihi tanggal akhir.")
        elif (end_date - start_date).days < 30: 
            st.warning("Rentang data terlalu pendek. Disarankan minimal 30 hari untuk analisis.")
        else:
            with st.spinner("Mempersiapkan analisis..."):
                # 1. Fetch Data
                st.write("1/4 - Menghubungi server Bank Indonesia...")
                df_raw = fetch_bi_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if df_raw.empty:
                    st.error("Tidak ada data ditemukan. Coba perluas rentang tanggal Anda.")
                    st.stop()

                # 2. Reshape Data
                st.write("2/4 - Membersihkan dan menyusun data...")
                df_long = reshape_and_clean_data(df_raw, details)
                if df_long.empty:
                    st.error("Data untuk komoditas terpilih tidak tersedia pada rentang waktu ini.")
                    st.stop()
               
                # 3. Feature Engineering
                st.write("3/4 - Menganalisis pola data historis...")
                sequence, feature_cols, error_msg = full_preparation_pipeline(df_long, details)
                if error_msg:
                    st.error(error_msg)
                    st.stop()

                # 4. Forecasting
                st.write("4/4 - Menghitung proyeksi 30 hari ke depan...")
                all_predicted_prices = forecast_iteratively(
                    models[selected_commodity], scalers[selected_commodity],
                    sequence, feature_cols, details['targets'], future_steps=30
                )
               
                forecast_dates = pd.date_range(start=datetime.now() + timedelta(days=1), periods=30)
                df_forecast = pd.DataFrame(all_predicted_prices, index=forecast_dates, columns=details['targets'])
                df_forecast.index.name = "Tanggal"

                st.session_state.results = {
                    "df_forecast": df_forecast,
                    "sequence": sequence,
                    "details": details
                }
                st.session_state.prediction_generated = True


st.title("📈 Proyeksi Harga Pangan Strategis Jawa Barat")
st.markdown("Gunakan aplikasi ini untuk menganalisis tren historis dan mendapatkan proyeksi harga komoditas pangan untuk 30 hari ke depan berdasarkan data dari PIHPS Nasional.")


with st.expander("ℹ️ Tentang Aplikasi & Data"):
    st.markdown("""
    - **Sumber Data**: Data harga diakses secara *real-time* dari **Pusat Informasi Harga Pangan Strategis (PIHPS) Nasional**, yang dikelola oleh Bank Indonesia.
    - **Sumber Pasar**: Semua data harga bersumber dari **pasar tradisional** di wilayah Jawa Barat.
    - **Satuan**: Beras & Telur Ayam (per kg), Minyak Goreng (per Liter).
   
    ***Disclaimer**: Proyeksi ini adalah hasil estimasi model matematis, bukan merupakan jaminan harga di masa depan.*
    """)



if st.session_state.prediction_generated and st.session_state.results:
    display_results(st.session_state.results)
else:
    st.markdown("---")
    st.info(
        "**Selamat Datang!** Silakan pilih parameter pada **panel di sebelah kiri (sidebar)** lalu klik tombol **'Cek Proyeksi Harga'** untuk memulai.",
        icon="👋"
    )

