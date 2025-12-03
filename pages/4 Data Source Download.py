import streamlit as st
import pandas as pd
from io import BytesIO
import os
from itertools import product

st.set_page_config(page_title="Download Data Source", page_icon="📥", layout="wide")
st.title("📥 Download Data Mentah per Stasiun & Variabel")

# --- KONFIGURASI DATA ---
# Ubah path ini sesuai dengan struktur direktori Anda
BASE_PATH = "data/5k_epoch/pred" 
dataset_options = {
    "0 Variabel": "0_var",
    "0 Variabel (NEW)": "0_var_new",
    "1 Variabel (W500_NEW)": "1_var_w500_new",
    "1 Variabel (W500_OLD)": "1_var_w500_old",
    "10 Variabel": "10_var",
    "10 Variabel (NEW)": "10_var_new",
    "51 Variabel": "51_var"
}

tahun_options = list(range(2010, 2015))
bulan_options = ["Januari","Februari","Maret","April","Mei","Juni",
                 "Juli","Agustus","September","Oktober","November","Desember"]

# --- PILIH DATASET & TANGGAL ---
dataset_name = st.selectbox("Pilih Dataset:", list(dataset_options.keys()))

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
def load_data(dataset_name, start_year, end_year):
    path_suffix = dataset_options[dataset_name]
    all_dfs = []
    
    # Memastikan tahun_end tidak lebih kecil dari tahun_start
    if start_year > end_year:
        st.warning("Tahun Awal harus kurang dari atau sama dengan Tahun Akhir.")
        return pd.DataFrame()

    for th in range(start_year, end_year + 1):
        file_path = os.path.join(BASE_PATH, path_suffix, f"all_data_{th}.parquet")
        
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

df = load_data(dataset_name, tahun_start, tahun_end)

# --- DEBUGGING BAGIAN PENTING ---
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
# Gunakan `stasiun_options` sebagai default jika ada
default_stasiun = stasiun_options[:3] if stasiun_options else []
selected_stasiun = st.multiselect("Pilih Stasiun:", stasiun_options, default=default_stasiun)

var_options = [c for c in df.columns if c not in ['stasiun','tahun']]
# Gunakan `var_options` sebagai default jika ada
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
    # Pastikan data yang dimasukkan adalah DataFrame
    df_input.to_excel(output, index=False)
    return output.getvalue()

st.info(f"Jumlah file yang akan di-download: **{len(selected_stasiun) * len(selected_vars)}**")

# --- DOWNLOAD FILE PER KOMBINASI ---
st.subheader("⬇️ Download Files")

# Pembatasan Bulan (jika diperlukan - saat ini tidak digunakan)
# Pastikan Anda memfilter DataFrame sesuai Bulan Awal/Akhir jika kolom tanggal/bulan ada

for stasiun_name, var_name in product(selected_stasiun, selected_vars):
    # Hanya mengambil kolom tahun, stasiun, dan variabel yang dipilih
    df_filtered = df[df['stasiun'] == stasiun_name][['tahun', 'stasiun', var_name]]
    
    file_name = f"{dataset_name.replace(' ', '_')}_{stasiun_name}_{var_name}_{tahun_start}-{tahun_end}.xlsx"
    
    # Konversi ke Excel
    excel_data = to_excel(df_filtered)
    
    st.download_button(
        label=f"⬇️ {stasiun_name} - {var_name}",
        data=excel_data,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"dl_{stasiun_name}_{var_name}" # Tambahkan key unik
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