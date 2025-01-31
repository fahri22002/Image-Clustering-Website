import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
from clustering import (
    process_image, initialize_centroids, kmeans_manual,
    predict_clustering, visualize_clusters, show_image_with_legend
)

# Fungsi untuk menghitung silhouette score
def calculate_silhouette_score(pixels, labels, sample_size=1000):
    if len(pixels) > sample_size:
        pixels_sampled, labels_sampled = resample(pixels, labels, n_samples=sample_size, random_state=42)
    else:
        pixels_sampled, labels_sampled = pixels, labels
    return silhouette_score(pixels_sampled, labels_sampled)

# Menambahkan tema
st.set_page_config(page_title="K-Means Clustering pada Gambar", layout="wide")

# Judul Aplikasi
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>K-Means Clustering pada Gambar</h1>", unsafe_allow_html=True)

# Deskripsi Aplikasi
st.markdown("""<p style='text-align: center; color: #666;'>Selamat datang di aplikasi K-Means Clustering.
Unggah gambar Anda untuk melatih model dan melakukan prediksi.</p>""", unsafe_allow_html=True)

# Pembagian Kolom untuk upload gambar
col1, col2 = st.columns(2)

# Input untuk mengunggah gambar pelatihan
with col1:
    st.subheader("Unggah Gambar untuk Training")
    train_images = st.file_uploader("Pilih gambar untuk folder Training", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

# Input untuk mengunggah gambar pengujian
with col2:
    st.subheader("Unggah Gambar untuk Testing")
    test_images = st.file_uploader("Pilih gambar untuk folder Testing", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

# Input untuk jumlah cluster
st.subheader("Parameter K-Means")
k = st.number_input("Masukkan jumlah cluster (k)", min_value=1, value=3)

# Tombol untuk melatih model
if st.button("Train"):
    with st.spinner('Tunggu, sedang training data...'):
        all_pixels = []
        for file in train_images:
            img = process_image(file)
            if img is not None:
                all_pixels.append(img)
        all_pixels = np.vstack(all_pixels)

        st.session_state.pixels = all_pixels
        st.session_state.centroids = initialize_centroids(all_pixels, k)

        # Proses training menggunakan K-Means manual
        st.session_state.centroids = kmeans_manual(st.session_state.pixels, k, st.session_state.centroids)
    
    st.success("Model berhasil dilatih!", icon="✅")

# Tombol untuk memprediksi
if st.button("Predict"):
    if 'pixels' not in st.session_state or 'centroids' not in st.session_state:
        st.warning("Silakan latih model terlebih dahulu.")
    else:
        st.subheader("Hasil Clustering")

        silhouette_scores = []

        for i, file in enumerate(test_images):
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixels = image.reshape(-1, 3)
            labels = predict_clustering(pixels, st.session_state.centroids)
            clustered_image = visualize_clusters(image, labels, k, st.session_state.centroids)

            # Hitung silhouette score
            score = calculate_silhouette_score(pixels, labels)
            silhouette_scores.append(score)

            # Hitung persentase tiap cluster
            unique, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100

            # Layout dengan dua kolom: kiri (gambar) dan kanan (pie chart)
            col_img, col_chart = st.columns([1, 1])

            with col_img:
                show_image_with_legend(clustered_image, st.session_state.centroids, k, f"Hasil Klustering {i + 1}")

            with col_chart:
                st.write(f"Silhouette Score: {score:.4f}")

                # Membuat pie chart untuk persentase cluster
                fig, ax = plt.subplots(figsize=(2, 2), dpi=150)

                # Warna sesuai centroid (diubah ke [0,1] untuk matplotlib)
                colors = [centroid / 255.0 for centroid in st.session_state.centroids]

                ax.pie(
                    percentages,
                    labels=[f"Cluster {j + 1}" for j in range(k)],
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90,
                    textprops={'fontsize': 8}
                )
                ax.axis('equal')  # Pie chart bulat

                # Tampilkan pie chart di Streamlit
                st.pyplot(fig)

            st.write("---")  # Garis pemisah antar gambar

# Menambahkan footer
st.markdown("""
<p style='text-align: center; color: #999;'>© 2024 K-Means Clustering</p>
<p style='text-align: center; color: #999;'>Tegar Muhamad Rizki &nbsp;&nbsp;&nbsp; Sanjukin Ndube Pinem &nbsp;&nbsp;&nbsp; Fahri Nizar Argubi</p>
""", unsafe_allow_html=True)



