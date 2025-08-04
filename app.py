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
    st.error(f"Gagal mengimpor modul dari 'src'. Pastikan struktur folder benar dan semua dependensi terinstal. Detail: {e}")
    st.stop()

st.set_page_config(
    layout="wide",
    page_title="Proyeksi Harga Pangan",
    page_icon="📈"
)

# =============================================================================
# HELPER
# =============================================================================

@st.cache_resource
def load_models_and_dependencies():
    try:
        models, scalers = load_all_models_and_scalers(COMMODITY_CONFIG)
        return models, scalers
    except Exception as e:
        st.error(f"Gagal memuat file model/scaler. Detail: {e}")
        st.stop()

models, scalers = load_models_and_dependencies()

def display_results():
    action = st.session_state.get('action', 'forecast')
    results = st.session_state.results
    
    title_map = {
        'forecast': "Hasil Proyeksi Harga", 'tren': "Hasil Analisis Tren", 'data': "Detail Data Harga"
    }
    st.header(title_map.get(action, "Hasil Analisis"))
    
    if st.button("↩️ Buat Analisis Baru"):
        for key in st.session_state.keys():
            if key not in ['models', 'scalers']:
                del st.session_state[key]
        st.rerun()
    st.markdown("---")

    df_history = results.get('df_history')
    df_forecast = results.get('df_forecast')
    details = results.get('details')

    def render_tab_ringkasan():
        st.subheader(f"Highlight Proyeksi untuk {details['main']}")
        tomorrow = datetime.now() + timedelta(days=1)
        days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
        months = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
        formatted_date = f"{days[tomorrow.weekday()]}, {tomorrow.day} {months[tomorrow.month]} {tomorrow.year}"
        st.markdown(f"Estimasi harga untuk esok hari: **{formatted_date}**.")
        st.write("") 
        cols = st.columns(len(details['targets']))
        next_day_prices = df_forecast.iloc[0]
        for i, target_name in enumerate(details['targets']):
            with cols[i]:
                last_known_price = df_history[target_name].iloc[-1]
                predicted_price = next_day_prices[target_name]
                delta_value = predicted_price - last_known_price
                st.markdown(f"**{target_name}**")
                st.markdown(f"<p style='font-size: 2rem; font-weight: 600; margin: 0;'>Rp {predicted_price:,.0f}</p>", unsafe_allow_html=True)
                if delta_value >= 0:
                    arrow, bg_color, text_color = "↑", "#ffeded", "#a60000"
                else:
                    arrow, bg_color, text_color = "↓", "#e6ffed", "#006400"
                delta_text = f"Rp {abs(delta_value):,.0f}"
                delta_html = f"""<div style="display: inline-block; background-color: {bg_color}; color: {text_color}; padding: 3px 8px; border-radius: 15px; font-size: 0.9em; font-weight: 500; line-height: 1;">{arrow} {delta_text}</div>"""
                st.markdown(delta_html, unsafe_allow_html=True)
        st.caption("Perubahan harga (delta) dibandingkan dengan harga historis terakhir yang diketahui.")

    def render_tab_grafik():
        st.subheader("Grafik Tren Harga")
        if df_forecast is None and action == 'forecast':
             st.info("💡 Grafik ini hanya menampilkan data historis karena data proyeksi gagal dibuat. Coba dengan rentang tanggal yang lebih panjang.", icon="ℹ️")
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i, col in enumerate(details['targets']):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=df_history.index, y=df_history[col], mode='lines+markers', name=f'Historis - {col}', line=dict(color=color)))
            if df_forecast is not None:
                fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast[col], mode='lines', name=f'Proyeksi - {col}', line=dict(dash='dash', color=color)))
                fig.add_vline(x=df_history.index[-1].value, line_width=2, line_dash="dot", line_color="grey", annotation_text="Mulai Proyeksi", annotation_position="top right")
        fig.update_layout(xaxis_title='Tanggal', yaxis_title='Harga (Rp)', hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    def render_tab_data():
        df_to_show = df_forecast if df_forecast is not None else df_history
        st.subheader("Tabel Detail Data")
        st.dataframe(df_to_show.style.format("Rp {:,.0f}"), use_container_width=True)
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=True).encode('utf-8')
        csv = convert_df_to_csv(df_to_show)
        st.download_button(label="📥 Download Data (CSV)", data=csv, file_name=f"data_{details['main']}_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

    def render_tab_statistik():
        st.subheader("Analisis Statistik Data Historis")
        st.markdown(f"Statistik dihitung dari {df_history.index.min().strftime('%d %B %Y')} hingga {df_history.index.max().strftime('%d %B %Y')}.")
        st.divider()
        stats_cols = st.columns(len(details['targets']))
        for i, target_name in enumerate(details['targets']):
            with stats_cols[i]:
                st.markdown(f"<h5>{target_name}</h5>", unsafe_allow_html=True)
                stats_data = {
                    "Harga Rata-rata": f"Rp {df_history[target_name].mean():,.0f}",
                    "Harga Tertinggi": f"Rp {df_history[target_name].max():,.0f}",
                    "Harga Terendah": f"Rp {df_history[target_name].min():,.0f}",
                    "Tingkat Fluktuasi (StDev)": f"Rp {df_history[target_name].std():,.0f}"
                }
                for label, value in stats_data.items():
                    st.caption(label)
                    st.markdown(f"**{value}**")
        st.divider()
        st.info(
            icon="💡",
            body="""
            **Apa itu Tingkat Fluktuasi (StDev)?**

            Standar Deviasi (StDev) mengukur seberapa besar harga suatu barang menyebar dari harga rata-ratanya. Semakin tinggi angkanya, semakin tidak stabil atau semakin **sering harga barang tersebut berfluktuasi** selama periode yang dipilih.
            """
        )

    if action == 'forecast':
        tabs = st.tabs(["📊 Ringkasan Proyeksi", "📈 Grafik Tren", "📋 Data Detail", "🔬 Analisis Statistik"])
        with tabs[0]:
            if df_forecast is not None and not df_forecast.empty:
                render_tab_ringkasan()
            else:
                st.warning("Gagal menghasilkan data proyeksi.", icon="⚠️")
                st.info("Penyebab umum: rentang data historis yang dipilih terlalu pendek atau tidak mengandung cukup variasi. Silakan coba lagi dengan rentang tanggal yang lebih panjang (disarankan > 90 hari).")
        with tabs[1]: render_tab_grafik()
        with tabs[2]: render_tab_data()
        with tabs[3]: render_tab_statistik()
    elif action == 'tren':
        tabs = st.tabs(["📈 Grafik Tren", "🔬 Analisis Statistik"])
        with tabs[0]: render_tab_grafik()
        with tabs[1]: render_tab_statistik()
    elif action == 'data':
        render_tab_data()

def show_homepage():
    col1, col2 = st.columns([1, 2.5], gap="large")
    with col1:
        st.markdown("<div style='display: flex; align-items: center; justify-content: center; height: 100%;'><p style='font-size: 8rem; text-align: center;'>💡</p></div>", unsafe_allow_html=True)
    with col2:
        st.subheader("Selamat Datang di Dashboard Analisis Harga Pangan")
        st.write("Platform ini membantu Anda memahami fluktuasi harga pangan di Jawa Barat. Anda dapat melihat tren historis, mengunduh data, hingga mendapatkan proyeksi harga berbasis *machine learning*.")
        st.write("Pilih salah satu menu di bawah ini untuk memulai. 👇🏻")
    st.divider()

    st.write("#### Apa yang ingin Anda lakukan?")
    col1, col2, col3 = st.columns(3, gap="medium")
    def set_action(action_type):
        st.session_state.view = 'show_form'
        st.session_state.action = action_type

    with col1:
        with st.container(border=True, height=240):
            st.markdown("<h3 style='text-align: center;'>📊 Analisis Tren</h3>", unsafe_allow_html=True)
            st.write("Memvisualisasikan data historis dalam grafik dan melihat statistik utamanya untuk memahami pola pasar.")
            st.button("Mulai Analisis", use_container_width=True, on_click=set_action, args=['tren'], key='b1')
    with col2:
        with st.container(border=True, height=240):
            st.markdown("<h3 style='text-align: center;'>🤖 Proyeksi Harga</h3>", unsafe_allow_html=True)
            st.write("Menggunakan model *Deep Learning* untuk mendapat estimasi harga komoditas hingga 30 hari ke depan.")
            st.button("Buat Proyeksi", type="primary", use_container_width=True, on_click=set_action, args=['forecast'], key='b2')
    with col3:
        with st.container(border=True, height=240):
            st.markdown("<h3 style='text-align: center;'>📋 Detail Data</h3>", unsafe_allow_html=True)
            st.write("Melihat dan mengunduh data harga dalam format tabel (CSV) untuk Anda olah dan analisis lebih lanjut.")
            st.button("Lihat & Unduh Data", use_container_width=True, on_click=set_action, args=['data'], key='b3')

def show_parameter_form():
    action = st.session_state.get('action', 'forecast')
    title_map = {'forecast': "Parameter Proyeksi Harga", 'tren': "Parameter Analisis Tren", 'data': "Parameter Data Detail"}
    st.header(f"⚙️ {title_map.get(action)}")

    selected_commodity = st.selectbox(
        "Pilih Kelompok Komoditas", 
        list(COMMODITY_CONFIG.keys()), 
        help="Pilih kelompok komoditas yang ingin dianalisis. Detail di bawah akan otomatis berubah."
    )
    details = COMMODITY_CONFIG[selected_commodity]
    with st.expander("Lihat Detail Sub-Komoditas"):
        for target in details['targets']:
            st.markdown(f"- {target}")

    with st.form(key="parameter_form"):
        st.info("Pilih rentang data historis sebagai dasar analisis. Disarankan minimal 90 hari untuk hasil proyeksi yang lebih akurat.", icon="💡")
        today = datetime.now().date()
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Dari Tanggal", value=today - timedelta(days=90), max_value=today - timedelta(days=1))
        with c2:
            end_date = st.date_input("Hingga Tanggal", value=today, max_value=today)
        st.divider()
        submitted = st.form_submit_button("Proses Data", type="primary", use_container_width=True)

        if submitted:
            if start_date >= end_date or (end_date - start_date).days < 7:
                st.error("Rentang tanggal tidak valid. Pastikan minimal 7 hari dan tanggal mulai sebelum tanggal akhir.")
            else:
                st.session_state.params = {"selected_commodity": selected_commodity, "start_date": start_date, "end_date": end_date}
                st.session_state.view = 'run_process'
                st.rerun()
    if st.button("Kembali ke Halaman Awal"):
        st.session_state.view = 'home'
        st.rerun()

def run_processing():
    params = st.session_state.params
    action = st.session_state.action
    details = COMMODITY_CONFIG[params["selected_commodity"]]
    with st.spinner("Memproses data..."):
        st.write("1/2 - Mengambil & membersihkan data historis...")
        df_raw = fetch_bi_data(params["start_date"].strftime('%Y-%m-%d'), params["end_date"].strftime('%Y-%m-%d'))
        if df_raw.empty: st.error("Tidak ada data ditemukan."); st.stop()
        df_long = reshape_and_clean_data(df_raw, details)
        if df_long.empty: st.error(f"Data untuk '{params['selected_commodity']}' tidak tersedia."); st.stop()
        sequence, feature_cols, error_msg = full_preparation_pipeline(df_long, details)
        if error_msg: st.error(error_msg); st.stop()
        df_history = sequence[details['targets']]
        df_forecast = None

        if action == 'forecast':
            st.write("2/2 - Menghitung proyeksi 30 hari ke depan...")
            if (params["end_date"] - params["start_date"]).days < 30:
                st.warning("Rentang data untuk proyeksi disarankan minimal 30 hari untuk hasil optimal.", icon="⚠️")
            all_predicted_prices = forecast_iteratively(models[params['selected_commodity']], scalers[params['selected_commodity']], sequence.copy(), feature_cols, details['targets'], future_steps=30)
            forecast_dates = pd.date_range(start=datetime.now().date() + timedelta(days=1), periods=30)
            df_forecast = pd.DataFrame(all_predicted_prices, index=forecast_dates, columns=details['targets'])
            df_forecast.index.name = "Tanggal"

        st.session_state.results = {"df_history": df_history, "df_forecast": df_forecast, "details": details}
        st.session_state.view = 'show_results'
        st.rerun()

# =============================================================================
# MAIN APP ROUTER
# =============================================================================

def main():
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>Analisis dan Proyeksi Harga Pangan</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Platform Cerdas untuk Memantau Fluktuasi Harga Pangan Strategis di Jawa Barat</p>", unsafe_allow_html=True)
    st.divider()

    if 'view' not in st.session_state:
        st.session_state.view = 'home'

    view_router = {'home': show_homepage, 'show_form': show_parameter_form, 'run_process': run_processing, 'show_results': display_results}
    view_router[st.session_state.view]()

    st.divider()
    with st.expander("ℹ️ Tentang Aplikasi, Sumber Data, & Disclaimer"):
        st.markdown("""
        - **Sumber Data**: Data harga diakses dari **Pusat Informasi Harga Pangan Strategis (PIHPS) Nasional** (Bank Indonesia).
        - **Model**: Proyeksi dihasilkan oleh model *Long Short-Term Memory* (LSTM).
        - ***Disclaimer***: Proyeksi ini adalah hasil estimasi model matematis dan bukan jaminan harga di masa depan.
        """)

if __name__ == "__main__":
    main()