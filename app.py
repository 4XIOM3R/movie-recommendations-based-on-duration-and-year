import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# LOAD DATASET
data = pd.read_csv('/content/data_cluster_netflix.csv')

# Pastikan kolom penting ini tersedia
required_cols = ['title', 'duration', 'release_year', 'rating']
missing = [c for c in required_cols if c not in data.columns]
if missing:
    st.error(f"Kolom berikut tidak ditemukan: {missing}")
    st.stop()

# Bersihkan kolom duration
def clean_duration(x):
    if isinstance(x, str):
        if 'min' in x:
            return int(x.replace(' min',''))
        elif 'Season' in x:
            return int(x.split()[0])
    return x

data['duration'] = pd.to_numeric(data['duration'].apply(clean_duration), errors='coerce')
data['release_year'] = pd.to_numeric(data['release_year'], errors='coerce')
data.dropna(subset=['duration','release_year'], inplace=True)

# Hapus outlier (IQR)
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

data = remove_outliers_iqr(data, 'duration')
data = remove_outliers_iqr(data, 'release_year')

#Mapping rating angka ke kategori jika masih numerik
rating_map = {
    0: "G", 1: "PG", 2: "PG-13", 3: "R", 4: "NC-17",
    5: "TV-Y", 6: "TV-Y7", 7: "TV-G", 8: "TV-PG",
    9: "TV-14", 10:"TV-MA"
}
if pd.api.types.is_numeric_dtype(data['rating']):
    data['rating'] = data['rating'].astype(int).map(rating_map)
data['rating'] = data['rating'].astype(str)


#FUNGSI REKOMENDASI
def rekomendasi_film(durasi, tahun, rating_filter, df, n_rekom=5):
    df_filtered = df.copy()
    if rating_filter != "Semua":
        df_filtered = df_filtered[df_filtered['rating'] == rating_filter]
    if df_filtered.empty:
        return pd.DataFrame(), "Tidak ada film dengan rating tersebut."

    fitur = df_filtered[['duration','release_year']].astype(float)
    user_input = pd.DataFrame([[durasi, tahun]], columns=['duration','release_year'], dtype=float)

    df_filtered = df_filtered.copy()
    df_filtered['similarity'] = cosine_similarity(user_input, fitur)[0]
    return df_filtered.sort_values(by='similarity', ascending=False).head(n_rekom), None

#STREAMLIT UI
st.set_page_config(page_title="Sistem Rekomendasi Film", layout="wide")
st.title("Sistem Rekomendasi Film Netflix")
st.markdown("Masukkan **Durasi**, **Tahun Rilis**, dan pilih **Rating** untuk menemukan film yang mirip!")

durasi_input = st.number_input("Durasi Film (menit):", min_value=1, max_value=500, value=120)
tahun_input = st.number_input("Tahun Rilis:", min_value=1900, max_value=2025, value=2020)
rating_list = ["Semua"] + sorted(data['rating'].unique().tolist())
rating_filter = st.selectbox("Pilih Rating Film:", rating_list)

#TOMBOL REKOMENDASI
if st.button("Cari Rekomendasi"):
    hasil, pesan = rekomendasi_film(durasi_input, tahun_input, rating_filter, data)
    if pesan:
        st.warning(pesan)
    else:
        st.success(f"Rekomendasi berdasarkan durasi **{durasi_input} menit**, tahun **{tahun_input}**, rating **{rating_filter}**:")
        st.dataframe(hasil[['title','duration','release_year','rating']])

# FOOTER
st.markdown("---")
st.caption("Dibangun dengan Streamlit | Dataset Clustering Netflix | Erlangga Wijaya 2025")


