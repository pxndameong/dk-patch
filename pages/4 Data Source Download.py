import streamlit as st
import pandas as pd
from io import BytesIO
import os
from itertools import product

# Menghilangkan warning Streamlit Watchdog
os.environ["STREAMLIT_WATCHDOG"] = "false"

st.set_page_config(page_title="Download Data Source", page_icon="📥", layout="wide")
st.title("📥 Download Data Mentah per Stasiun & Variabel")

# --- DEKLARASI URL DASAR BARU DAN INFO DATASET ---
# URL Dasar Utama (sesuaikan jika path root data/5k_epoch/pred berubah)
BASE_PATH_PRED_ROOT = "data/5k_epoch/pred" 
base_url_padanan = "data/5k_epoch/padanan" # Tetap sama

# Info dataset yang akan dibandingkan
dataset_info = {
    # Menyimpan path Suffix yang sesuai dengan folder dan Prefix nama file yang dikonfirmasi
    "0 Variabel": {"path_suffix": "0_var", "prefix": "all_data_0var"},
    "0 Variabel (NEW)": {"path_suffix": "0_var_new", "prefix": "all_data_0var"},
    "1 Variabel (W500_NEW)": {"path_suffix": "1_var_w500_new", "prefix": "all_data_1var"},
    "1 Variabel (W500_OLD)": {"path_suffix": "1_var_w500_old", "prefix": "all_data_1var_w500"}, 
    "10 Variabel": {"path_suffix": "10_var", "prefix": "all_data_10var"},
    "10 Variabel (NEW)": {"path_suffix": "10_var_new", "prefix": "all_data_10var"},
    "51 Variabel": {"path_suffix": "51_var", "prefix": "all_data_51var"},
}
dataset_options = list(dataset_info.keys()) # Untuk st.selectbox

tahun_options = list(range(2010, 2015))
bulan_options = ["Januari","Februari","Maret","April","Mei","Juni",
                 "Juli","Agustus","September","Oktober","November","Desember"]
bulan_dict = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}

# --- PILIH DATASET & TANGGAL ---
dataset_name = st.selectbox("Pilih Dataset:", dataset_options)
selected_dataset_config = dataset_info[dataset_name]

col1, col2, col3, col4 = st.columns(4)
with col1:
    bulan_start = st.selectbox("Bulan Awal:", bulan_options, index=0)
with col2:
    tahun_start = st.selectbox("Tahun Awal:", tahun_options, index=0)
with col3:
    bulan_end = st.selectbox("Bulan Akhir:", bulan_options, index=11)
with col4:
    tahun_end = st.selectbox("Tahun Akhir:", tahun_options, index=len(tahun_options)-1)

# --- LOAD DATA ---
@st.cache_data
def load_data(config, start_year, end_year):
    # Mengambil path_suffix dan prefix dari konfigurasi yang dipilih
    path_suffix = config["path_suffix"]
    prefix = config["prefix"] 
    
    all_dfs = []
    
    if start_year > end_year:
        st.warning("Tahun Awal harus kurang dari atau sama dengan Tahun Akhir.")
        return pd.DataFrame()

    for th in range(start_year, end_year + 1):
        # *** PERBAIKAN PENTING DI SINI ***
        # Menggunakan prefix dari dataset_info untuk menyusun nama file
        file_name = f"{prefix}_{th}.parquet"
        file_path = os.path.join(BASE_PATH_PRED_ROOT, path_suffix, file_name)
        
        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                df["tahun"] = th
                all_dfs.append(df)
            except Exception as e:
                st.error(f"Gagal membaca file {file_path}: {e}")
        else:
            st.info(f"File tidak ditemukan: {file_path}") # Memberi tahu user file mana yang hilang
            
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

df = load_data(selected_dataset_config, tahun_start, tahun_end)

# --- DEBUGGING STATUS PEMUATAN DATA ---
st.markdown("---")
st.subheader("💡 Status Pemuatan Data")
if df.empty:
    st.error("Data tidak ditemukan untuk pilihan Dataset & Rentang Tahun ini.")
    st.stop()
else:
    st.success(f"Data berhasil dimuat. Total baris: {len(df):,}")
    st.markdown(f"Kolom yang tersedia: `{', '.join(df.columns)}`")
st.markdown("---")

# --- MULTI-SELECT STASIUN & VARIABEL ---
# Cek Kolom 'stasiun'
if 'stasiun' not in df.columns:
    st.error("Kolom 'stasiun' tidak ada di data. Tidak dapat melanjutkan.")
    st.stop()

stasiun_options = df['stasiun'].unique().tolist()
default_stasiun = stasiun_options[:3] if stasiun_options else []
selected_stasiun = st.multiselect("Pilih Stasiun:", stasiun_options, default=default_stasiun)

var_options = [c for c in df.columns if c not in ['stasiun','tahun']]
default_vars = var_options[:3] if var_options else []
selected_vars = st.multiselect("Pilih Variabel:", var_options, default=default_vars)

# --- CEK PILIHAN KOSONG ---
if not selected_stasiun:
    st.warning("Silakan pilih minimal 1 stasiun.")
    st.stop()
if not selected_vars:
    st.warning("Silakan pilih minimal 1 variabel.")
    st.stop()

# --- KONVERSI EXCEL ---
def to_excel(df_input):
    output = BytesIO()
    df_input.to_excel(output, index=False)
    return output.getvalue()

st.info(f"Jumlah file yang akan di-download: **{len(selected_stasiun) * len(selected_vars)}**")

# --- DOWNLOAD FILE PER KOMBINASI ---
st.subheader("⬇️ Download Files")

for stasiun_name, var_name in product(selected_stasiun, selected_vars):
    df_filtered = df[df['stasiun'] == stasiun_name][['tahun', 'stasiun', var_name]]
    
    # Menggunakan nama dataset yang lebih bersih untuk nama file
    dataset_clean_name = dataset_name.replace(' ', '_').replace('(', '').replace(')', '')
    file_name = f"{dataset_clean_name}_{stasiun_name}_{var_name}_{tahun_start}-{tahun_end}.xlsx"
    
    excel_data = to_excel(df_filtered)
    
    st.download_button(
        label=f"⬇️ {stasiun_name} - {var_name}",
        data=excel_data,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"dl_{stasiun_name}_{var_name}"
    )


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
    margin-top: 40px;
    }
    </style>

    <div class="fade-in-text">
    <h4>BRIN Research Team</h4>
    <p><em>Data Visualization by Tsaqib</em></p>
    </div>
    """,
    unsafe_allow_html=True
)