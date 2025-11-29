import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# =====================
# 1. Load Dataset
# =====================
data_rekomendasi = pd.read_csv('/content/databaru_cluster.csv')

# Pastikan kolom utama ada
required_cols = ['title', 'duration', 'release_year', 'rating']
missing = [c for c in required_cols if c not in data_rekomendasi.columns]
if missing:
    st.error(f"Kolom berikut tidak ditemukan dalam dataset: {missing}")
    st.stop()

# Bersihkan data
data_rekomendasi.dropna(subset=required_cols, inplace=True)

# =====================
# 2. Fungsi Rekomendasi
# =====================
def rekomendasi_film_manual(durasi_input, tahun_input, rating_filter, data, n_rekom=5):
    # Filter data berdasarkan rating
    if rating_filter != "Semua":
        data = data[data['rating'] == rating_filter]

    if data.empty:
        return pd.DataFrame(), "Tidak ada film dengan rating tersebut."

    # Ambil fitur numerik
    fitur = data[['duration', 'release_year']]
    user_input = pd.DataFrame([[durasi_input, tahun_input]], columns=['duration', 'release_year'])

    # Hitung kemiripan
    sim = cosine_similarity(user_input, fitur)[0]
    data['similarity'] = sim

    # Urutkan hasil
    rekom = data.sort_values(by='similarity', ascending=False).head(n_rekom)
    return rekom, None

# =====================
# 3. Streamlit Layout
# =====================
st.set_page_config(page_title="🎬 Sistem Rekomendasi Film", layout="wide")
st.title("🎥 Sistem Rekomendasi Film Netflix")
st.markdown("Masukkan **Durasi**, **Tahun Rilis**, dan pilih **Rating** untuk menemukan film yang mirip!")

# Input user
durasi_input = st.number_input("Masukkan Durasi Film (menit):", min_value=1, max_value=500, value=120)
tahun_input = st.number_input("Masukkan Tahun Rilis:", min_value=1900, max_value=2025, value=2020)
rating_list = ["Semua"] + sorted(data_rekomendasi['rating'].dropna().unique().tolist())
rating_filter = st.selectbox("Pilih Rating Film:", rating_list)

# Tombol rekomendasi
if st.button("Cari Rekomendasi"):
    hasil, pesan = rekomendasi_film_manual(durasi_input, tahun_input, rating_filter, data_rekomendasi)

    if pesan:
        st.warning(pesan)
    else:
        st.success(f"film yang mirip dengan durasi **{durasi_input} menit**, tahun **{tahun_input}**, rating **{rating_filter}**:")
        st.dataframe(hasil[['title', 'duration', 'release_year', 'rating']])

        # =====================
        # 4. Visualisasi
        # =====================
        st.subheader("Visualisasi (Durasi vs Tahun Rilis)")
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(data_rekomendasi['duration'], data_rekomendasi['release_year'],
                             alpha=0.4, color='gray', label='Semua Film')

        if not hasil.empty:
            ax.scatter(hasil['duration'], hasil['release_year'], color='orange', s=100, label='Rekomendasi')

        ax.scatter(durasi_input, tahun_input, color='red', s=200, label='Input Anda', edgecolor='black')
        ax.set_xlabel("Durasi (menit)")
        ax.set_ylabel("Tahun Rilis")
        ax.legend()
        st.pyplot(fig)

st.markdown("---")
st.caption("Dibangun dengan Streamlit | Dataset Clustering Netflix | Erlangga Wijaya")
