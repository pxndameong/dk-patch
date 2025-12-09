# pages/Analytical Table.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Tabel Data Komparatif",
    page_icon="📊",
    layout="wide"
)

# Menghilangkan warning Streamlit Watchdog
os.environ["STREAMLIT_WATCHDOG"] = "false"

# --- DEKLARASI URL DASAR BARU DAN INFO DATASET ---
# URL Dasar Utama (sesuaikan jika path root data/5k_epoch/pred berubah)
# Asumsi path root adalah: 'data/5k_epoch/pred'
BASE_PATH_PRED_ROOT = "data/5k_epoch/pred"
base_url_padanan = "data9sta/5k_epoch/padanan" # Tetap sama
##
# Info dataset yang akan dibandingkan (Menggabungkan Standar dan W500)
dataset_info = {
    # Menyimpan path Suffix yang sesuai dengan folder yang dikonfirmasi
    "0 Variabel": {"path_suffix": "0_var", "prefix": "all_data_0var"},
    "1 Variabel": {"path_suffix": "1_var", "prefix": "all_data_1var"},
    "10 Variabel": {"path_suffix": "10_var", "prefix": "all_data_10var"},
    "51 Variabel": {"path_suffix": "51_var", "prefix": "all_data_51var"},

}

bulan_dict = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}

# Data stasiun yang baru
station_data = [
    {"name": "Stasiun 218 (Lat: -8, Lon: 113.5)", "lat": -8.0, "lon": 113.5, "index": 218},
    {"name": "Stasiun 294 (Lat: -7.5, Lon: 110)", "lat": -7.5, "lon": 110.0, "index": 294},
    {"name": "Stasiun 329 (Lat: -7.25, Lon: 107.5)", "lat": -7.25, "lon": 107.5, "index": 329},
    {"name": "Stasiun 384 (Lat: -7, Lon: 110)", "lat": -7.0, "lon": 110.0, "index": 384},
    {"name": "Stasiun 393 (Lat: -7, Lon: 112.25)", "lat": -7.0, "lon": 112.25, "index": 393},
    {"name": "Stasiun 505 (Lat: -6.25, Lon: 106.5)", "lat": -6.25, "lon": 106.5, "index": 505},
    {"name": "Stasiun 417 (Lat: -6.75, Lon: 107)", "lat": -6.75, "lon": 107.0, "index": 417},
    {"name": "Stasiun 168 (Lat: -8.25, Lon: 112.25)", "lat": -8.25, "lon": 112.25, "index": 168},
    {"name": "Stasiun 256 (Lat: -7.75, Lon: 111.75)", "lat": -7.75, "lon": 111.75, "index": 256},
]

# MODIFIKASI 1: Hapus opsi "Rata-Rata Seluruh Stasiun" dari daftar
station_names = [s["name"] for s in station_data]
# station_names.insert(0, "Rata-Rata Seluruh Stasiun") # Baris ini dihapus

# Precompute station coords set for fast filtering (tuples of floats)
station_coords = {(s["lat"], s["lon"]) for s in station_data}

@st.cache_data
def load_data(dataset_name: str, tahun: int):
    # --- LOGIKA PENENTUAN BASE URL YANG KONSOLIDASI ---
    path_suffix = dataset_info[dataset_name]["path_suffix"]
    prefix = dataset_info[dataset_name]["prefix"]
    
    # Path penuh: BASE_PATH_PRED_ROOT / path_suffix / prefix_tahun.parquet
    # Contoh: data/5k_epoch/pred/1_var_w500_new/all_data_1var_2010.parquet
    url = f"{BASE_PATH_PRED_ROOT}/{path_suffix}/{prefix}_{tahun}.parquet" 
    
    try:
        df = pd.read_parquet(url, engine="pyarrow")
    except Exception as e:
        # st.error(f"DEBUG GAGAL LOAD {dataset_name} ({url}): {e}") 
        return pd.DataFrame()
    
    df = df.convert_dtypes()
    
    # JAMIN KEBERADAAN KOLOM KOORDINAT UNTUK DATA PREDIKSI
    # 1. LATITUDE
    if 'lat' in df.columns and 'latitude' not in df.columns:
        df = df.rename(columns={'lat': 'latitude'})
    elif 'latitude' not in df.columns:
        df['latitude'] = np.nan 

    # 2. LONGITUDE
    if 'lon' in df.columns and 'longitude' not in df.columns:
        df = df.rename(columns={'lon': 'longitude'})
    elif 'longitude' not in df.columns:
        df['longitude'] = np.nan 
    
    if 'latitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    if 'longitude' in df.columns:
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
            
    return df

@st.cache_data
def load_padanan_data(tahun: int):
    """
    Fungsi untuk memuat data padanan.
    """
    url = f"{base_url_padanan}/CLEANED_PADANAN_{tahun}.parquet"
    required_cols = ['month', 'year', 'latitude', 'longitude', 'rainfall', 'idx']

    try:
        df = pd.read_parquet(url, engine="pyarrow")
    except Exception as e:
        # st.warning(f"⚠️ Gagal membaca file padanan: {url}\nError: {e}") 
        return pd.DataFrame(columns=required_cols) 
    
    # Logika Renaming
    if 'lon' in df.columns:
        df = df = df.rename(columns={'lon': 'longitude'})
    if 'lat' in df.columns:
        df = df.rename(columns={'lat': 'latitude'})
    if 'idx_new' in df.columns:
        df = df.rename(columns={'idx_new': 'idx'})

    # JAMIN KEBERADAAN required_cols SEBELUM SELECTION
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan 

    if 'latitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    if 'longitude' in df.columns:
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    return df[required_cols]

# --- FUNGSI UNTUK MENGHITUNG METRIK ---
def calculate_metrics(df: pd.DataFrame, actual_col: str, pred_col: str):
    """
    Menghitung MAE, RMSE, dan R^2.
    Menggunakan kolom yang tersedia dan hanya baris tanpa NaN di kedua kolom.
    """
    df_clean = df.dropna(subset=[actual_col, pred_col])

    if df_clean.empty:
        return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}

    actual = df_clean[actual_col].astype(float)
    pred = df_clean[pred_col].astype(float)

    mae = np.mean(np.abs(pred - actual))
    rmse = np.sqrt(np.mean((pred - actual)**2))

    ss_total = np.sum((actual - np.mean(actual))**2)
    ss_residual = np.sum((actual - pred)**2)

    if ss_total == 0:
        r2 = 1.0 
    else:
        r2 = 1 - (ss_residual / ss_total)

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# MODIFIKASI: Menerima list of station names
def plot_comparative_charts_monthly(tahun_start: int, bulan_start: int, tahun_end: int, bulan_end: int, selected_station_names: list):
    """
    Fungsi untuk menampilkan bar chart perbandingan curah hujan bulanan (prediksi vs ground truth)
    dan di bawahnya, Scatter Plot MAE, RMSE, dan R^2 dalam rentang waktu kustom.
    """
    
    # LOGIKA PENENTUAN LABEL JUDUL BERDASARKAN SELEKSI
    if not selected_station_names:
        st.error("❌ Pilih setidaknya satu stasiun untuk ditampilkan.")
        return
        
    # MODIFIKASI: Karena opsi 'Rata-Rata Seluruh Stasiun' dihapus, is_all_stations didefinisikan sebagai True 
    # hanya jika SEMUA stasiun yang tersedia dipilih, atau jika ada lebih dari 1 stasiun.
    is_all_stations_selected = set(selected_station_names) == set(station_names)
    is_multiple_stations = len(selected_station_names) > 1

    if is_all_stations_selected:
        display_name = "Rata-Rata Seluruh Stasiun"
        coords_to_filter = station_coords
    elif is_multiple_stations:
        display_name = f"Rata-Rata {len(selected_station_names)} Stasiun Dipilih"
        coords_to_filter = {
            (s['lat'], s['lon']) for s in station_data if s['name'] in selected_station_names
        }
    else: # Hanya 1 stasiun dipilih
        display_name = selected_station_names[0]
        station_info = next((s for s in station_data if s["name"] == selected_station_names[0]), None)
        coords_to_filter = {(station_info['lat'], station_info['lon'])} if station_info else set()

    if not coords_to_filter:
         st.error("❌ Informasi koordinat stasiun tidak ditemukan.")
         return

    st.markdown("---")

    start_date_str = f"{bulan_dict[bulan_start]} {tahun_start}"
    end_date_str = f"{bulan_dict[bulan_end]} {tahun_end}"
    date_range_str = f"{start_date_str} - {end_date_str}"

    st.subheader(f"Perbandingan Curah Hujan Bulanan dan Metrik ({date_range_str}) untuk {display_name}")

    all_data_for_plot = []
    all_combined_data = {} 

    years_to_load = list(range(tahun_start, tahun_end + 1))

    # --- Pemuatan Data Padanan (Ground Truth) ---
    df_padanan_all = []
    for th in years_to_load:
        df_padanan_all.append(load_padanan_data(th))
        
    df_padanan_all = [df for df in df_padanan_all if not df.empty]
    
    if len(df_padanan_all) == 0:
        st.error("❌ Tidak ada file padanan ditemukan untuk tahun yang dipilih.")
        return

    df_padanan_full = pd.concat(df_padanan_all, ignore_index=True)

    start_date = pd.to_datetime(f"{tahun_start}-{bulan_start}-01")
    end_date = pd.to_datetime(f"{tahun_end}-{bulan_end}-01")

    # 1. Filter/Rata-rata Ground Truth
    df_padanan_stations = df_padanan_full.copy()
    
    if {'latitude', 'longitude'}.issubset(df_padanan_stations.columns):
        df_padanan_stations['latitude'] = df_padanan_stations['latitude'].astype(float)
        df_padanan_stations['longitude'] = df_padanan_stations['longitude'].astype(float)
        df_padanan_stations['coord_tuple'] = list(zip(df_padanan_stations['latitude'], df_padanan_stations['longitude']))
        
        # FILTER MENGGUNAKAN COORDS_TO_FILTER
        df_padanan_stations = df_padanan_stations[df_padanan_stations['coord_tuple'].isin(coords_to_filter)].copy()
        df_padanan_stations.drop(columns=['coord_tuple'], inplace=True, errors='ignore')
    else:
        df_padanan_stations = pd.DataFrame(columns=df_padanan_full.columns)

    if df_padanan_stations.empty:
        st.warning(f"⚠️ Ground Truth (padanan) untuk stasiun yang dipilih tidak ditemukan dalam data padanan.")
        df_padanan_filtered = pd.DataFrame()
    else:
        df_padanan_stations['date'] = pd.to_datetime(df_padanan_stations[['year', 'month']].assign(day=1))
        df_padanan_stations = df_padanan_stations[(df_padanan_stations['date'] >= start_date) & (df_padanan_stations['date'] <= end_date)].copy()
        df_padanan_stations.drop(columns=['date'], inplace=True, errors='ignore')
        # SELALU GROUPBY DAN AGG UNTUK MENDAPATKAN RATA-RATA BULANAN
        df_padanan_filtered = df_padanan_stations.groupby(['year', 'month']).agg(rainfall=('rainfall', 'mean')).reset_index()

    df_padanan_station = df_padanan_filtered.copy() if isinstance(df_padanan_filtered, pd.DataFrame) else pd.DataFrame()

    if not df_padanan_station.empty and 'rainfall' in df_padanan_station.columns:
        df_padanan_plot = df_padanan_station.rename(columns={'rainfall': 'Curah Hujan (mm)'})
        df_padanan_plot['Tipe Data'] = 'Ground Truth (Rainfall)'
        all_data_for_plot.append(df_padanan_plot[['year', 'month', 'Curah Hujan (mm)', 'Tipe Data']])
    else:
        if df_padanan_station.empty:
            st.warning("⚠️ Ground Truth (Rainfall) tidak tersedia untuk stasiun/rata-rata ini di tahun yang dipilih.")


    # 2. Ambil data Prediksi (Iterasi di semua model)
    for dataset_name in dataset_info.keys():
        df_pred_all = []
        for th in years_to_load:
            df_pred_all.append(load_data(dataset_name, th))
            
        df_pred_all = [df for df in df_pred_all if not df.empty]
        
        if len(df_pred_all) == 0:
            df_pred_full = pd.DataFrame()
        else:
            df_pred_full = pd.concat(df_pred_all, ignore_index=True)

        df_pred_station = pd.DataFrame() 

        # *** PENCEGAHAN ERROR PADA DF KOSONG ***
        if df_pred_full.empty:
             st.warning(f"⚠️ Data Prediksi ({dataset_name}) tidak ditemukan atau kosong untuk rentang waktu yang dipilih.")
             all_combined_data[dataset_name] = pd.DataFrame()
             continue 

        # Filter/Rata-rata Prediksi
        if {'latitude', 'longitude'}.issubset(df_pred_full.columns):
            df_pred_full['latitude'] = pd.to_numeric(df_pred_full['latitude'], errors='coerce')
            df_pred_full['longitude'] = pd.to_numeric(df_pred_full['longitude'], errors='coerce')
            
            df_pred_full['coord_tuple'] = list(zip(df_pred_full['latitude'].astype(float), df_pred_full['longitude'].astype(float)))
            
            # FILTER MENGGUNAKAN COORDS_TO_FILTER
            df_pred_stations = df_pred_full[df_pred_full['coord_tuple'].isin(coords_to_filter)].copy()
            df_pred_stations.drop(columns=['coord_tuple'], inplace=True, errors='ignore')
        else:
            df_pred_stations = pd.DataFrame(columns=df_pred_full.columns)

        if df_pred_stations.empty:
            st.warning(f"⚠️ Prediksi ({dataset_name}) tidak memiliki grid yang cocok dengan stasiun yang dipilih.")
            df_pred_filtered = pd.DataFrame()
        else:
            df_pred_stations['date'] = pd.to_datetime(df_pred_stations[['year', 'month']].assign(day=1))
            df_pred_stations = df_pred_stations[(df_pred_stations['date'] >= start_date) & (df_pred_stations['date'] <= end_date)].copy()
            df_pred_stations.drop(columns=['date'], inplace=True, errors='ignore')
            # SELALU GROUPBY DAN AGG UNTUK MENDAPATKAN RATA-RATA BULANAN
            df_pred_filtered = df_pred_stations.groupby(['year', 'month']).agg(ch_pred=('ch_pred', 'mean')).reset_index()

        df_pred_station = df_pred_filtered.copy() if isinstance(df_pred_filtered, pd.DataFrame) else pd.DataFrame()

        # --- PENANGANAN MERGE ---
        required_pred_cols = ['year', 'month', 'ch_pred']

        if not df_pred_station.empty and all(col in df_pred_station.columns for col in required_pred_cols):
            # Gabungkan data Prediksi dan Aktual (Padanan)
            df_merged_custom_range = pd.merge(
                df_pred_station[required_pred_cols],
                df_padanan_station[['year', 'month', 'rainfall']],
                on=['year', 'month'],
                how='inner'
            ).drop_duplicates(subset=['year', 'month'])

            all_combined_data[dataset_name] = df_merged_custom_range

            # Data untuk Bar Chart
            df_pred_plot = df_pred_station.rename(columns={'ch_pred': 'Curah Hujan (mm)'})
            df_pred_plot['Tipe Data'] = f'Prediksi ({dataset_name})'
            all_data_for_plot.append(df_pred_plot[['year', 'month', 'Curah Hujan (mm)', 'Tipe Data']])

        else:
            all_combined_data[dataset_name] = pd.DataFrame()


    # --- Plot Bar Chart (Plotly Express) ---
    if not all_data_for_plot:
        st.error("❌ Tidak ada data (prediksi maupun ground truth) yang ditemukan untuk periode ini.")
        return

    df_plot = pd.concat(all_data_for_plot, ignore_index=True)
    df_plot['Bulan-Tahun'] = df_plot['month'].map(bulan_dict) + ' ' + df_plot['year'].astype(str)

    df_plot['date_sort'] = pd.to_datetime(df_plot[['year', 'month']].assign(day=1))
    df_plot = df_plot.sort_values(by=['date_sort', 'Tipe Data'])

    bar_color_map = {
        'Ground Truth (Rainfall)': 'saddlebrown',
        'Prediksi (0 Variabel)': 'royalblue',
        'Prediksi (1 Variabel)': 'mediumvioletred',
        'Prediksi (10 Variabel)': 'deeppink',
        'Prediksi (51 Variabel)': 'forestgreen'
    }

    x_order = df_plot['Bulan-Tahun'].unique().tolist()

    fig_bar = px.bar(
        df_plot,
        x='Bulan-Tahun',
        y='Curah Hujan (mm)',
        color='Tipe Data',
        barmode='group',
        color_discrete_map=bar_color_map,
        title=f'Curah Hujan Bulanan Komparatif ({date_range_str}) di {display_name}',
        labels={'Curah Hujan (mm)': 'Curah Hujan (mm)', 'Bulan-Tahun': 'Bulan-Tahun'},
    )

    fig_bar.update_layout(
        xaxis_title="Bulan-Tahun",
        yaxis_title="Curah Hujan (mm)",
        legend_title="Tipe Data",
        bargap=0.15,
        xaxis={'categoryorder': 'array', 'categoryarray': x_order}
    )

    st.plotly_chart(fig_bar, use_container_width=True)
    # --- Akhir Plot Bar Chart ---

    st.markdown("---")
    st.subheader(f"Scatter Plot Curah Hujan Aktual vs Prediksi Bulanan ({date_range_str})")

    # --- Plot Scatter Plot (Matplotlib) ---
    model_data_to_plot = {k: v for k, v in all_combined_data.items() if not v.empty}
    num_models = len(model_data_to_plot)
    
    if num_models == 0:
        st.error("❌ Tidak ada data prediksi yang cukup untuk membuat Scatter Plot.")
        return

    # Penyesuaian layout subplot agar tidak kosong
    if num_models <= 4:
        cols = num_models if num_models > 0 else 1
        rows = 1
        figsize = (5 * cols, 5)
    else:
        cols = 4
        rows = int(np.ceil(num_models / cols))
        figsize = (20, 5 * rows)
    
    fig_scatter, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() 
    
    # Nonaktifkan subplot yang tidak terpakai
    for j in range(num_models, len(axes)):
        fig_scatter.delaxes(axes[j])
        
    plt.style.use('ggplot')

    scatter_color_map = {
        '0 Variabel': 'royalblue',
        '0 Variabel (NEW)': 'navy',
        '1 Variabel (W500_NEW)': 'mediumvioletred',
        '1 Variabel (W500_OLD)': 'darkorange',
        '10 Variabel': 'deeppink',
        '10 Variabel (NEW)': 'crimson',
        '51 Variabel': 'forestgreen'
    }

    i = 0
    max_val = 0
    
    for dataset_name, df_combined in model_data_to_plot.items():
        ax = axes[i]
            
        metrics = calculate_metrics(df_combined, 'rainfall', 'ch_pred')

        actual = df_combined['rainfall'].astype(float)
        pred = df_combined['ch_pred'].astype(float)

        if not actual.empty and not pred.empty:
            current_max = max(actual.max(), pred.max())
            if current_max > max_val:
                max_val = current_max

        ax.scatter(actual, pred, color=scatter_color_map.get(dataset_name, 'gray'), label=dataset_name, alpha=0.7)

        textstr = '\n'.join((
            r'MAE = %.2f' % (metrics['MAE'], ),
            r'RMSE = %.2f' % (metrics['RMSE'], ),
            r'$R^2$ = %.2f' % (metrics['R2'], )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.0)
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)

        ax.set_title(f'Model {dataset_name}', fontsize=12)
        ax.set_xlabel('Aktual Bulanan (mm)', fontsize=10)
        # HANYA TAMPILKAN YLABEL DI KOLOM PERTAMA
        if i % cols == 0: 
            ax.set_ylabel('Prediksi Bulanan (mm)', fontsize=10)
        
        i += 1

    plot_limit = max_val * 1.05 if max_val > 0 else 100
    
    # Terapkan batas sumbu yang seragam
    for j in range(num_models):
        axes[j].set_xlim(0, plot_limit)
        axes[j].set_ylim(0, plot_limit)
        axes[j].plot([0, plot_limit], [0, plot_limit], color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    st.pyplot(fig_scatter)
# --- Akhir Plot Scatter Plot ---

# Main Streamlit app logic (TETAP SAMA)
st.title("📊 DK Viewer - Tabel Analisis Komparatif")

if 'comparative_data' not in st.session_state:
    st.session_state.comparative_data = None
if 'combinations' not in st.session_state:
    st.session_state.combinations = []
# MODIFIKASI 2: selected_station_names diinisialisasi sebagai list kosong
if 'selected_station_names' not in st.session_state:
    st.session_state.selected_station_names = station_names # Default ke semua stasiun

with st.sidebar.form("config_form"):
    st.header("⚙️ Konfigurasi")

    year_options = list(range(2010, 2015))
    bulan_options = list(range(1, 13))

    display_type = st.radio(
        "Pilih Tampilan Data:",
        ["Time Series and Summary", "Bar Chart and Metrics"]
    )

    # --- Konfigurasi Rentang Waktu ---
    st.subheader("Dari")
    col1, col2 = st.columns(2)
    with col1:
        bulan_from = st.selectbox(
            "Bulan Awal:",
            bulan_options,
            index=0,
            format_func=lambda x: bulan_dict[x],
            key="bulan_from_ts" if display_type == "Time Series and Summary" else "bulan_from_bar"
        )
    with col2:
        tahun_from = st.selectbox("Tahun Awal:", year_options, index=0, key="tahun_from_ts" if display_type == "Time Series and Summary" else "tahun_from_bar")


    st.subheader("Sampai")
    col3, col4 = st.columns(2)
    with col3:
        bulan_until = st.selectbox(
            "Bulan Akhir:",
            bulan_options,
            index=len(bulan_options) - 1,
            format_func=lambda x: bulan_dict[x],
            key="bulan_until_ts" if display_type == "Time Series and Summary" else "bulan_until_bar"
        )
    with col4:
        tahun_until = st.selectbox("Tahun Akhir:", year_options, index=len(year_options) - 1, key="tahun_until_ts" if display_type == "Time Series and Summary" else "tahun_until_bar")

    # MODIFIKASI 3: Menggunakan st.multiselect dengan default=station_names (SELECT ALL)
    selected_station_names = st.multiselect("Pilih stasiun:", station_names, default=station_names)
    st.session_state.selected_station_names = selected_station_names

    submit = st.form_submit_button("🚀 Submit konfigurasi dan bandingkan")

if submit:
    from_date_tuple = (tahun_from, bulan_from)
    until_date_tuple = (tahun_until, bulan_until)

    if from_date_tuple > until_date_tuple:
        st.error("❌ Tanggal 'Dari' tidak boleh lebih baru dari tanggal 'Sampai'.")
    elif not selected_station_names:
        st.error("❌ Pilih setidaknya satu stasiun.")
    elif display_type == "Bar Chart and Metrics":
        # Gunakan list nama stasiun di sini
        st.session_state.comparative_data = {} 
        plot_comparative_charts_monthly(tahun_from, bulan_from, tahun_until, bulan_until, selected_station_names)
        
        # Penentuan label untuk pesan sukses
        is_all_stations_selected = set(selected_station_names) == set(station_names)
        display_name_msg = "Rata-Rata Seluruh Stasiun" if is_all_stations_selected else (selected_station_names[0] if len(selected_station_names) == 1 else f"Rata-Rata {len(selected_station_names)} Stasiun Dipilih")
        st.success(f"✅ Data berhasil dimuat dan siap untuk Bar Chart dan Scatter Plot untuk rentang **{bulan_dict[bulan_from]} {tahun_from}** hingga **{bulan_dict[bulan_until]} {tahun_until}** untuk {display_name_msg}.")

    else: # Time Series and Summary
        tahun_final = list(range(tahun_from, tahun_until + 1))
        
        # MODIFIKASI: Logika penentuan koordinat yang dipilih
        selected_coords = {
            (s['lat'], s['lon']) for s in station_data if s['name'] in selected_station_names
        }

        filtered_data_dict = {}

        for dataset_name in dataset_info.keys():
            all_filtered = []

            for th in tahun_final:
                df_main = load_data(dataset_name, th)
                df_padanan = load_padanan_data(th)
                
                if not df_main.empty and not df_padanan.empty:
                    # Gabungkan dengan padanan
                    df_merged_year = pd.merge(df_main, df_padanan, on=['month', 'year', 'latitude', 'longitude'], how='left', suffixes=('_pred', '_actual'))
                    if 'ch_pred_pred' in df_merged_year.columns:
                        df_merged_year = df_merged_year.rename(columns={'ch_pred_pred': 'ch_pred'})
                    
                    df_merged_year = df_merged_year.drop_duplicates(subset=['latitude', 'longitude', 'month', 'year'])
                    all_filtered.append(df_merged_year)
                elif not df_main.empty:
                    all_filtered.append(df_main)


            if all_filtered:
                df_filtered_all = pd.concat(all_filtered, ignore_index=True)
                df_temp = df_filtered_all.copy()
                
                # Logika filter dan agregasi yang dipersatukan
                if {'latitude', 'longitude'}.issubset(df_temp.columns):
                    df_temp['latitude'] = pd.to_numeric(df_temp['latitude'], errors='coerce')
                    df_temp['longitude'] = pd.to_numeric(df_temp['longitude'], errors='coerce')
                    df_temp['coord_tuple'] = list(zip(df_temp['latitude'].astype(float), df_temp['longitude'].astype(float)))
                    
                    # FILTER BERDASARKAN STASIUN YANG DIPILIH
                    if selected_coords and df_temp['coord_tuple'].isin(selected_coords).any():
                        df_temp = df_temp[df_temp['coord_tuple'].isin(selected_coords)].copy()
                    else:
                        df_temp = pd.DataFrame() # Data stasiun tidak ditemukan
                        
                    if 'coord_tuple' in df_temp.columns:
                        df_temp.drop(columns=['coord_tuple'], inplace=True, errors='ignore')
                
                # AGGREGASI RATA-RATA BULANAN
                aggregation_cols = ['ch_pred']
                if 'rainfall' in df_temp.columns:
                    aggregation_cols.append('rainfall')

                if not df_temp.empty and 'ch_pred' in df_temp.columns:
                    df_filtered_station = df_temp.groupby(['year', 'month']).agg(
                        **{col: (col, 'mean') for col in aggregation_cols}
                    ).reset_index()
                else:
                    df_filtered_station = pd.DataFrame()
                    
                # PERHITUNGAN METRIK
                if 'rainfall' in df_filtered_station.columns and 'ch_pred' in df_filtered_station.columns and not df_filtered_station.empty:
                    df_filtered_station['error_bias'] = df_filtered_station['ch_pred'].astype(float) - df_filtered_station['rainfall'].astype(float)
                    df_filtered_station['absolute_error'] = abs(df_filtered_station['ch_pred'].astype(float) - df_filtered_station['rainfall'].astype(float))
                    df_filtered_station['squared_error'] = (df_filtered_station['ch_pred'].astype(float) - df_filtered_station['rainfall'].astype(float))**2
                else:
                    df_filtered_station['error_bias'] = np.nan
                    df_filtered_station['absolute_error'] = np.nan
                    df_filtered_station['squared_error'] = np.nan

                if not df_filtered_station.empty:
                    # FILTER RENTANG TANGGAL
                    mask = (
                        (df_filtered_station['year'] > tahun_from) |
                        ((df_filtered_station['year'] == tahun_from) & (df_filtered_station['month'] >= bulan_from))
                    ) & (
                        (df_filtered_station['year'] < tahun_until) |
                        ((df_filtered_station['year'] == tahun_until) & (df_filtered_station['month'] <= bulan_until))
                    )
                    df_filtered_station = df_filtered_station[mask].copy()
                else:
                    df_filtered_station = pd.DataFrame()

                filtered_data_dict[dataset_name] = df_filtered_station
            else:
                filtered_data_dict[dataset_name] = pd.DataFrame()

        st.session_state.comparative_data = filtered_data_dict
        
        is_all_stations_selected = set(selected_station_names) == set(station_names)
        display_name_msg = "Rata-Rata Seluruh Stasiun" if is_all_stations_selected else (selected_station_names[0] if len(selected_station_names) == 1 else f"Rata-Rata {len(selected_station_names)} Stasiun Dipilih")
        st.success(f"✅ Data berhasil dimuat dan siap untuk perbandingan Time Series dari **{bulan_dict[bulan_from]} {tahun_from}** hingga **{bulan_dict[bulan_until]} {tahun_until}** untuk {display_name_msg}.")

# --- Tampilan Time Series and Summary ---
if st.session_state.comparative_data and st.session_state.comparative_data.keys() and display_type == "Time Series and Summary":
    
    # Penentuan label stasiun
    selected_names = st.session_state.selected_station_names
    is_all_stations_selected = set(selected_names) == set(station_names)
    if is_all_stations_selected:
        current_station_name = "Rata-Rata Seluruh Stasiun"
    elif len(selected_names) == 1:
        current_station_name = selected_names[0]
    elif len(selected_names) > 1:
        current_station_name = f"Rata-Rata {len(selected_names)} Stasiun Dipilih"
    else:
        current_station_name = "Stasiun Tidak Dipilih"
        
    st.markdown("---")
    st.subheader(f"Ringkasan Statistik Komparatif untuk {current_station_name}")

    summary_cols = ['ch_pred', 'rainfall', 'error_bias', 'absolute_error', 'squared_error']
    comparison_summary = []

    for dataset_name, df in st.session_state.comparative_data.items():
        if not df.empty:
            summary_row = {"Metrik": dataset_name}
            # Cek kolom 'rainfall' (Ground Truth) hanya untuk model prediksi
            is_valid_comparison = 'rainfall' in df.columns and pd.notna(df['rainfall']).any()
            
            for col in summary_cols:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    # Untuk metrik error, hanya hitung jika perbandingan valid
                    if col in ['error_bias', 'absolute_error', 'squared_error'] and not is_valid_comparison:
                        summary_row[f"Mean ({col})"] = None
                        summary_row[f"Sum ({col})"] = None
                        continue
                    
                    summary_row[f"Mean ({col})"] = df[col].mean()
                    summary_row[f"Sum ({col})"] = df[col].sum()
                else:
                    summary_row[f"Mean ({col})"] = None
                    summary_row[f"Sum ({col})"] = None
            comparison_summary.append(summary_row)

    if comparison_summary:
        comparison_df = pd.DataFrame(comparison_summary).set_index("Metrik").T
        for col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else None)

        st.dataframe(comparison_df)
    else:
        st.warning("⚠️ Tidak ada data untuk ditampilkan. Pastikan rentang waktu valid.")

    st.markdown("---")
    st.subheader("Plot Perbandingan Time Series")

    selected_models = st.multiselect(
        "Pilih model yang akan di-plot:",
        options=list(dataset_info.keys()),
        default=list(dataset_info.keys()),
        key='ts_models'
    )

    metrics_to_plot = st.multiselect(
        "Pilih metrik untuk di-plot:",
        options=['ch_pred', 'rainfall', 'error_bias', 'absolute_error', 'squared_error'],
        default=['ch_pred', 'rainfall'],
        key='ts_metrics'
    )

    if not selected_models or not metrics_to_plot:
        st.info("💡 Pilih setidaknya satu model dan satu metrik untuk menampilkan plot.")
    else:
        is_rainfall_selected = 'rainfall' in metrics_to_plot
        other_metrics = [m for m in metrics_to_plot if m != 'rainfall']

        dfs_to_plot = []

        # Logika untuk Ground Truth (Rainfall)
        if is_rainfall_selected and selected_models:
            first_model = selected_models[0]
            df_rainfall = st.session_state.comparative_data.get(first_model)
            if df_rainfall is not None and not df_rainfall.empty and 'rainfall' in df_rainfall.columns:
                rainfall_df = df_rainfall[['year', 'month', 'rainfall']].copy()
                rainfall_df['model_name'] = 'Ground Truth'
                rainfall_df = rainfall_df.rename(columns={'rainfall': 'Value'})
                rainfall_df['Metric'] = 'rainfall'
                dfs_to_plot.append(rainfall_df)

        # Logika untuk Prediksi dan Error Metrik
        for model_name in selected_models:
            df = st.session_state.comparative_data.get(model_name, pd.DataFrame())
            if not df.empty:
                existing_other_metrics = [m for m in other_metrics if m in df.columns]

                # Pastikan 'ch_pred' termasuk jika dipilih
                if 'ch_pred' in metrics_to_plot and 'ch_pred' in df.columns and 'ch_pred' not in existing_other_metrics:
                     existing_other_metrics.append('ch_pred') 

                if existing_other_metrics:
                    cols_to_select = ['year', 'month'] + list(set(existing_other_metrics))
                    cols_to_select = [col for col in cols_to_select if col in df.columns]
                    
                    df_other_metrics = df[cols_to_select].copy()
                    df_other_metrics['model_name'] = model_name

                    melted_df = df_other_metrics.melt(
                        id_vars=['year', 'month', 'model_name'],
                        value_vars=list(set(existing_other_metrics)),
                        var_name='Metric',
                        value_name='Value'
                    )
                    dfs_to_plot.append(melted_df)


        if not dfs_to_plot:
            st.warning("⚠️ Data tidak tersedia untuk model atau metrik yang dipilih.")
        else:
            combined_df = pd.concat(dfs_to_plot, ignore_index=True).drop_duplicates(subset=['year', 'month', 'model_name', 'Metric'])
            combined_df['date'] = pd.to_datetime(combined_df[['year', 'month']].assign(day=1))
            combined_df.sort_values(by='date', inplace=True)
            combined_df['combined_label'] = combined_df['Metric'] + ' (' + combined_df['model_name'] + ')'
            combined_df.loc[combined_df['Metric'] == 'rainfall', 'combined_label'] = 'Rainfall (Ground Truth)'

            color_map = {
                'Rainfall (Ground Truth)': 'saddlebrown', 
                'ch_pred (0 Variabel)': 'royalblue',
                'error_bias (0 Variabel)': 'darkblue', 
                'absolute_error (0 Variabel)': 'midnightblue',
                'squared_error (0 Variabel)': 'navy', 
                
                'ch_pred (1 Variabel)': 'mediumvioletred',
                'error_bias (1 Variabel)': 'darkmagenta', 
                'absolute_error (1 Variabel)': 'purple',
                'squared_error (1 Variabel)': 'darkslategray',
                
                'ch_pred (10 Variabel)': 'deeppink',
                'error_bias (10 Variabel)': 'darkred', 
                'absolute_error (10 Variabel)': 'crimson',
                'squared_error (10 Variabel)': 'indianred', 

                
                'ch_pred (51 Variabel)': 'forestgreen',
                'error_bias (51 Variabel)': 'darkgreen', 
                'absolute_error (51 Variabel)': 'seagreen',
                'squared_error (51 Variabel)': 'olivedrab',
            }

            fig = px.line(
                combined_df,
                x='date', y='Value', color='combined_label',
                title=f'Perbandingan Time Series untuk {current_station_name}',
                labels={'Value': 'Nilai', 'date': 'Tanggal', 'combined_label': 'Metrik'},
                markers=True, color_discrete_map=color_map
            )
            st.plotly_chart(fig, use_container_width=True)

# --- Pesan Akhir ---
elif not submit:
    st.info("💡 Pilih Tampilan Data, rentang waktu/tahun, dan stasiun di sidebar, lalu tekan 'Submit konfigurasi dan bandingkan' untuk melihat data.")

st.markdown(
    """
    <style>
    @keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
    }
    .fade-in-text {
    animation: fadeIn 2s ease-in-out;
    text-align: center;
    margin-top: 20px;
    }
    </style>

    <div class="fade-in-text">
    <h4>BRIN Research Team</h4>
    <p><em>Data Visualization by Tsaqib</em></p>
    </div>
    """,
    unsafe_allow_html=True
)