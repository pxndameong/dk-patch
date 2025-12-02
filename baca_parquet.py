import pandas as pd

# Path file parquet
file_path = r"C:\Users\HP Elitebook X360\vscode\TSAQIB\pxnda.venv\Kode Pxnda\DK\dk-viewer-5k-1var\data\5k_epoch\padanan\CLEANED_PADANAN_1985.parquet"

# Baca parquet
df = pd.read_parquet(file_path)

# Tampilkan 50 baris pertama
print(df.head(200))

# Kalau ingin tahu jumlah baris & kolom
print("\nShape:", df.shape)
