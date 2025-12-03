import streamlit as st
import pandas as pd
from io import BytesIO
import os
from itertools import product

# Menghilangkan warning Streamlit Watchdog
os.environ["STREAMLIT_WATCHDOG"] = "false"

st.set_page_config(page_title="Download Data Source", page_icon="📥", layout="wide")
st.title("📥 Download Data Mentah per Stasiun & Variabel")

# --- 1. KONFIGURASI PATH & DATASET ---
BASE_PATH_PRED_ROOT = "data/5k_epoch/pred" 

# Info dataset yang akan dibandingkan (Pastikan prefix sesuai dengan nama file Anda)
dataset_info = {
    "0 Variabel": {"path_suffix": "0_var", "prefix": "all_data_0var"},
    "0 Variabel (NEW)": {"path_suffix": "0_var_new", "prefix": "all_data_0var"},
    "1 Variabel (W500_NEW)": {"path_suffix": "1_var_w500_new", "prefix": "all_data_1var"},
    "1 Variabel (W500_OLD)": {"path_suffix": "1_var_w500_old", "prefix": "all_data_1var_w500"}, 
    "10 Variabel": {"path_suffix": "10_var", "prefix": "all_data_10var"},
    "10 Variabel (NEW)": {"path_suffix": "10_var_new", "prefix": "all_data_10var"},
    "51 Variabel": {"path_suffix": "51_var", "prefix": "all_data_51var"},
}
dataset_options = list(dataset_info.keys())

tahun_options = list(range(2010, 2015))
bulan_options = ["Januari","Februari","Maret","April","Mei","Juni",
                 "Juli","Agustus","September","Oktober","November","Desember"]

# --- 2. PILIH DATASET & RENTANG TAHUN ---
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

# --- 3. FUNGSI LOAD DATA ---
@st.cache_data
def load_data(config, start_year, end_year):
    path_suffix = config["path_suffix"]
    prefix = config["prefix"] 
    all_dfs = []
    
    if start_year > end_year:
        st.warning("Tahun Awal harus kurang dari atau sama dengan Tahun Akhir.")
        return pd.DataFrame()

    for th in range(start_year, end_year + 1):
        # Menyusun nama file menggunakan prefix yang benar
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
            st.info(f"File tidak ditemukan: {file_path}")
            
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

df = load_data(selected_dataset_config, tahun_start, tahun_end)

# --- 4. DEBUGGING DAN PERBAIKAN NAMA KOLOM ---
st.markdown("---")
st.subheader("💡 Status Pemuatan Data")

if df.empty:
    st.error("Data tidak ditemukan untuk pilihan Dataset & Rentang Tahun ini.")
    st.stop()

# --- >>> PERHATIAN: UBAH INI JIKA NAMA KOLOM STASIUN ANDA BERBEDA <<< ---
# Contoh: Jika di file Parquet Anda bernama 'Station_ID', ganti string di bawah.
# Jika kolom sudah bernama 'stasiun', biarkan nilainya None (atau ganti dengan 'stasiun')
NAMA_KOLOM_STASIUN_ASLI = None # Ganti dengan nama kolom yang sebenarnya, misal: 'station_id'
TARGET_NAMA = 'stasiun'

if NAMA_KOLOM_STASIUN_ASLI and NAMA_KOLOM_STASIUN_ASLI in df.columns:
    df = df.rename(columns={NAMA_KOLOM_STASIUN_ASLI: TARGET_NAMA})
    st.warning(f"Kolom '{NAMA_KOLOM_STASIUN_ASLI}' berhasil diganti namanya menjadi '{TARGET_NAMA}'.")
elif TARGET_NAMA not in df.columns:
    # Jika 'stasiun' tetap tidak ditemukan
    st.error(f"Kolom '{TARGET_NAMA}' tidak ditemukan di data.")
    st.warning(f"Kolom yang tersedia di data: {df.columns.tolist()}")
    st.stop()
# --- <<< AKHIR PERHATIAN >>> ---

st.success(f"Data berhasil dimuat. Total baris: {len(df):,}")
st.markdown(f"Kolom yang digunakan: `{', '.join(df.columns)}`")
st.markdown("---")


# --- 5. MULTI-SELECT STASIUN & VARIABEL ---
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

# --- 6. FUNGSI KONVERSI EXCEL & DOWNLOAD ---
def to_excel(df_input):
    output = BytesIO()
    df_input.to_excel(output, index=False)
    return output.getvalue()

st.info(f"Jumlah file yang akan di-download: **{len(selected_stasiun) * len(selected_vars)}**")

st.subheader("⬇️ Download Files")

for stasiun_name, var_name in product(selected_stasiun, selected_vars):
    df_filtered = df[df['stasiun'] == stasiun_name][['tahun', 'stasiun', var_name]]
    
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