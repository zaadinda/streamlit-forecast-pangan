import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

try:
    from src.config import COMMODITY_CONFIG
    from src.data_handler import fetch_bi_data, reshape_and_clean_data
    from src.feature_engineering import full_preparation_pipeline
    from src.predictions import load_all_models_and_scalers, forecast_iteratively
except ImportError as e:
    st.error(f"Gagal mengimpor modul dari 'src'. Pastikan struktur folder benar. Detail: {e}")
    st.stop()


st.set_page_config(
    layout="wide",
    page_title="Proyeksi Harga Pangan",
    page_icon="📈"
)

def load_custom_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"File CSS '{file_name}' tidak ditemukan. Beberapa elemen mungkin tidak tertata dengan baik.")

@st.cache_resource
def load_models_and_dependencies():
    """Memuat model dan scaler sekali saja menggunakan cache resource Streamlit."""
    try:
        models, scalers = load_all_models_and_scalers(COMMODITY_CONFIG)
        return models, scalers
    except Exception as e:
        st.error(f"Gagal memuat file model/scaler. Pastikan path di 'src/config.py' benar. Detail: {e}")
        st.stop()

load_custom_css("style.css")
models, scalers = load_models_and_dependencies()


# =============================================================================
# 3. UI HELPER
# =============================================================================

def display_prediction_results(results: dict):
    """Fungsi terpusat untuk menampilkan semua hasil prediksi dalam layout tab."""
    st.success("✅ Proyeksi berhasil dibuat!")
    st.markdown("---")

    df_forecast = results['df_forecast']
    sequence_history = results['sequence_history']
    details = results['details']

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Ringkasan Proyeksi", 
        "📈 Grafik Tren", 
        "📋 Data Detail", 
        "🔬 Analisis Statistik"
    ])

    with tab1:
        st.subheader(f"Highlights Proyeksi untuk {details['main']}")
        
        tomorrow = datetime.now() + timedelta(days=1)
        days_in_indonesian = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
        months_in_indonesian = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
        formatted_date = f"{days_in_indonesian[tomorrow.weekday()]}, {tomorrow.day} {months_in_indonesian[tomorrow.month]} {tomorrow.year}"
        
        st.markdown(f"Berikut adalah estimasi harga untuk esok hari: **{formatted_date}**.")
        
        st.markdown('<div class="metric-grid-container">', unsafe_allow_html=True)
        next_day_prices = df_forecast.iloc[0]
        history_for_delta = sequence_history[details['targets']]

        for target_name in details['targets']:
            with st.container():
                last_known_price = history_for_delta[target_name].iloc[-1]
                predicted_price = next_day_prices[target_name]
                delta_value = predicted_price - last_known_price
                
                st.metric(
                    label=target_name,
                    value=f"Rp {predicted_price:,.0f}"
                )

                if delta_value >= 0:
                    arrow = "▲"
                    color = "red" 
                else:
                    arrow = "▼"
                    color = "green"
                
                delta_text = f"Rp {abs(delta_value):,.0f} vs kemarin"

                st.markdown(f'<p style="text-align: center; color:{color}; font-size: 0.9rem;">{arrow} {delta_text}</p>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Perubahan harga dibandingkan dengan harga historis terakhir yang diketahui.")

    with tab2:
        st.subheader("Grafik Tren Harga Historis vs. Harga Proyeksi 30 Hari")

        history_to_plot = sequence_history[details['targets']]
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i, col in enumerate(details['targets']):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=history_to_plot.index, y=history_to_plot[col], mode='lines', name=f'Historis - {col}', line=dict(color=color)))
            fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast[col], mode='lines', name=f'Proyeksi - {col}', line=dict(dash='dash', color=color)))
        fig.add_vline(x=history_to_plot.index[-1].value, line_width=2, line_dash="dot", line_color="grey", annotation_text="Mulai Proyeksi", annotation_position="top right")
        fig.update_layout(
            margin=dict(t=120, b=80),
            xaxis_title='Tanggal', yaxis_title='Harga (Rp)', hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Tabel Detail Proyeksi Harga 30 Hari")
        st.dataframe(df_forecast.style.format("Rp {:,.2f}"), use_container_width=True)
        @st.cache_data
        def convert_df_to_csv(df_to_convert):
            return df_to_convert.to_csv(index=True).encode('utf-8')
        csv = convert_df_to_csv(df_forecast)
        st.download_button(label="📥 Download Data Proyeksi (CSV)", data=csv,
           file_name=f"proyeksi_{details['main']}_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        
    with tab4:
        st.subheader(f"Analisis Statistik untuk {details['main']}")
        st.markdown(f"Statistik dihitung berdasarkan data historis yang Anda pilih ({sequence_history.index.min().strftime('%d %B %Y')} hingga {sequence_history.index.max().strftime('%d %B %Y')}).")
        
        history_for_stats = sequence_history[details['targets']]
        stats_cols = st.columns(len(details['targets']))

        for i, target_name in enumerate(details['targets']):
            with stats_cols[i]:
                st.markdown(f"#### {target_name}")
                st.markdown("---")
                
                stats_data = {
                    "Harga Rata-rata": f"Rp {history_for_stats[target_name].mean():,.0f}",
                    "Harga Tertinggi": f"Rp {history_for_stats[target_name].max():,.0f}",
                    "Harga Terendah": f"Rp {history_for_stats[target_name].min():,.0f}",
                    "Tingkat Fluktuasi (StDev)": f"Rp {history_for_stats[target_name].std():,.0f}"
                }
                
                for label, value in stats_data.items():
                    st.caption(label)
                    st.markdown(f"**{value}**")

        st.markdown("---",)
        st.info("**Apa itu Tingkat Fluktuasi (StDev)?**\n\nStandar Deviasi (StDev) mengukur seberapa besar harga suatu barang menyebar dari harga rata-ratanya. Semakin tinggi angkanya, semakin tidak stabil atau semakin **sering harga barang tersebut berfluktuasi** selama periode yang dipilih.", icon="💡")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    if 'prediction_generated' not in st.session_state:
        st.session_state.prediction_generated = False
        st.session_state.results = None

    with st.sidebar:
        st.title("⚙️ Parameter Proyeksi")
        st.markdown("Atur parameter di bawah ini untuk menghasilkan proyeksi.")
        selected_commodity = st.selectbox("Pilih Kelompok Komoditas", list(COMMODITY_CONFIG.keys()))
        details = COMMODITY_CONFIG[selected_commodity]
        with st.expander("Lihat Detail Sub-Komoditas"):
            for target in details['targets']:
                st.markdown(f"- {target}")
        st.subheader("🗓️ Periode Analisis")
        st.info("Pilih rentang data historis sebagai dasar analisis. Disarankan minimal 60-90 hari.", icon="💡")
        today = datetime.now().date()
        default_start_date = today - timedelta(days=90)
        start_date = st.date_input("Dari Tanggal", value=default_start_date, max_value=today)
        end_date = st.date_input("Hingga Tanggal", value=today, max_value=today)
        st.markdown("---")
        predict_button = st.button("💰 Cek Proyeksi Harga", type="primary", use_container_width=True)

    st.title("📈 Proyeksi Harga Pangan Strategis Jawa Barat")
    st.markdown("📣 Antisipasi fluktuasi harga! Lihat tren terkini dan dapatkan proyeksi harga pangan di Jawa Barat untuk 30 hari mendatang.")

    with st.expander("ℹ️ Tentang Aplikasi & Data"):
        st.markdown("""
        - **Sumber Data**: Data harga diakses secara *real-time* dari **Pusat Informasi Harga Pangan Strategis (PIHPS) Nasional**, yang dikelola oleh Bank Indonesia.
        - **Sumber Pasar**: Semua data harga bersumber dari **pasar tradisional** di wilayah Jawa Barat.
        - **Satuan**: Beras & Telur Ayam (per kg), Minyak Goreng (per Liter).
        
        ***‼️Disclaimer**: Proyeksi ini adalah hasil estimasi model matematis dan bukan merupakan jaminan harga di masa depan.*
        """)

    if predict_button:
        if start_date > end_date:
            st.error("Tanggal mulai tidak boleh melebihi tanggal akhir."); st.stop()
        if (end_date - start_date).days < 30:
            st.warning("Rentang data terlalu pendek. Disarankan minimal 30 hari untuk analisis."); st.stop()
        with st.spinner("Memproses data... Ini mungkin memakan waktu beberapa saat."):
            st.write("1/4 - Menghubungi server Bank Indonesia...")
            df_raw = fetch_bi_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if df_raw.empty: st.error("Tidak ada data ditemukan."); st.stop()
            st.write("2/4 - Membersihkan dan menyusun data...")
            df_long = reshape_and_clean_data(df_raw, details)
            if df_long.empty: st.error(f"Data untuk '{selected_commodity}' tidak tersedia."); st.stop()
            st.write("3/4 - Menganalisis pola data historis...")
            sequence, feature_cols, error_msg = full_preparation_pipeline(df_long, details)
            if error_msg: st.error(error_msg); st.stop()
            st.write("4/4 - Menghitung proyeksi 30 hari ke depan...")
            all_predicted_prices = forecast_iteratively(models[selected_commodity], scalers[selected_commodity], sequence, feature_cols, details['targets'], future_steps=30)
            forecast_dates = pd.date_range(start=datetime.now() + timedelta(days=1), periods=30)
            df_forecast = pd.DataFrame(all_predicted_prices, index=forecast_dates, columns=details['targets'])
            df_forecast.index.name = "Tanggal"
            st.session_state.results = {"df_forecast": df_forecast, "sequence_history": sequence, "details": details}
            st.session_state.prediction_generated = True
            st.rerun()

    if st.session_state.prediction_generated and st.session_state.results:
        display_prediction_results(st.session_state.results)
    else:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("📊 Analisis Tren")
            st.write("Lihat data harga historis dari PIHPS Nasional dalam grafik yang interaktif dan informatif.")
        with col2:
            st.subheader("🤖 Smart Forecasting")
            st.write("Dapatkan proyeksi harga untuk 30 hari ke depan menggunakan model Deep Learning LSTM dengan akurasi terbaik.")
        with col3:
            st.subheader("💡 Keputusan Terinformasi")
            st.write("Gunakan data, analisis fluktuasi, dan proyeksi untuk membantu strategi Anda.")
        st.info("**Selamat Datang!** Silakan pilih parameter pada panel di sebelah kiri untuk memulai.", icon="👋")


# =============================================================================
# 5. ENTRY POINT APLIKASI
# =============================================================================

if __name__ == "__main__":
    main()