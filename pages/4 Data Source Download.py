import streamlit as st
import pandas as pd
from io import BytesIO
import os
from itertools import product

st.set_page_config(page_title="Download Data Source", page_icon="📥", layout="wide")
st.title("📥 Download Data Mentah per Stasiun & Variabel")

# --- KONFIGURASI DATA ---
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

# --- PILIH DATASET ---
dataset_name = st.selectbox("Pilih Dataset:", list(dataset_options.keys()))

# --- RENTANG BULAN & TAHUN ---
col1, col2 = st.columns(2)
with col1:
    bulan_start = st.selectbox("Bulan Awal:", bulan_options, index=0)
with col2:
    tahun_start = st.selectbox("Tahun Awal:", tahun_options, index=0)

col3, col4 = st.columns(2)
with col3:
    bulan_end = st.selectbox("Bulan Akhir:", bulan_options, index=11)
with col4:
    tahun_end = st.selectbox("Tahun Akhir:", tahun_options, index=len(tahun_options)-1)

# --- LOAD DATA ---
@st.cache_data
def load_data(dataset_name, tahun_start, tahun_end):
    path_suffix = dataset_options[dataset_name]
    all_dfs = []
    for th in range(tahun_start, tahun_end+1):
        file_path = f"{BASE_PATH}/{path_suffix}/all_data_{th}.parquet"
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            df["tahun"] = th
            all_dfs.append(df)
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

df = load_data(dataset_name, tahun_start, tahun_end)
if df.empty:
    st.warning("Data tidak ditemukan untuk pilihan ini.")
    st.stop()

# --- MULTI-SELECT STASIUN & VARIABEL ---
stasiun_options = df['stasiun'].unique() if 'stasiun' in df.columns else []
selected_stasiun = st.multiselect("Pilih Stasiun:", stasiun_options, default=stasiun_options[:3])

var_options = [c for c in df.columns if c not in ['stasiun','tahun']]
selected_vars = st.multiselect("Pilih Variabel:", var_options, default=var_options[:3])

if not selected_stasiun or not selected_vars:
    st.warning("Silakan pilih minimal 1 stasiun dan 1 variabel.")
    st.stop()

# --- KONVERSI EXCEL ---
def to_excel(df):
    output = BytesIO()
    df.to_excel(output, index=False)
    return output.getvalue()

st.write(f"Jumlah file yang akan di-download: {len(selected_stasiun) * len(selected_vars)}")

# --- DOWNLOAD FILE PER KOMBINASI ---
for stasiun_name, var_name in product(selected_stasiun, selected_vars):
    df_filtered = df[df['stasiun']==stasiun_name][['tahun','stasiun',var_name]]
    file_name = f"{dataset_name}_{stasiun_name}_{var_name}_{tahun_start}-{tahun_end}.xlsx"
    excel_data = to_excel(df_filtered)
    st.download_button(
        label=f"⬇️ {stasiun_name} - {var_name}",
        data=excel_data,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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