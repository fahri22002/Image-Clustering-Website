import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

PATH_TRAIN = 'data'
PATH_TEST = 'test'
k = 3

# Fungsi untuk membaca dan memproses satu gambar
def process_image(uploaded_file, image_size=(75, 75)):
    # Membaca file gambar dari streamlit
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    if image is None:
        return None
    # Resize gambar
    image = cv2.resize(image, image_size)
    # Convert BGR ke RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Reshape menjadi 2D array (1 baris per pixel, 3 kolom untuk channel RGB)
    pixels = image.reshape(-1, 3)
    return pixels

# Path ke folder data
folder_path = PATH_TRAIN

def show_image(image, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# List untuk menyimpan semua pixel dari semua gambar
all_pixels = []

def euclidean_distance(point1, point2):
    """Hitung jarak Euclidean antara dua titik."""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def kmeans_manual(features, k, centroids, max_iters=100):
    for it in range(max_iters):
        # 2. Assign cluster
        labels = np.zeros(features.shape[0])
        for i in range(features.shape[0]):
            distances = np.array([euclidean_distance(features[i], centroid) for centroid in centroids])
            labels[i] = np.argmin(distances)  
        # 3. Update centroid
        new_centroids = np.zeros(centroids.shape)
        for j in range(k):
            if np.any(labels == j):  
                new_centroids[j] = features[labels == j].mean(axis=0)
        print(it)
        # 4. Periksa konvergensi (jika centroid tidak berubah)
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids
    return centroids

def initialize_centroids(data, k):
    """
    Memilih centroid awal untuk KMeans menggunakan metode KMeans++.

    Parameters:
    data : array-like, shape (n_samples, n_features)
        Data input.
    k : int
        Jumlah cluster.

    Returns:
    centroids : array, shape (k, n_features)
        Centroid awal untuk KMeans.
    """
    # Inisialisasi list untuk centroid
    centroids = []
    
    # Pilih centroid pertama secara acak dari data
    centroids.append(data[np.random.randint(0, len(data))])
    
    # Pilih centroid lainnya menggunakan probabilitas jarak
    for _ in range(1, k):
        # Hitung jarak terdekat dari tiap data point ke centroid yang sudah ada
        distances = np.array([min([np.linalg.norm(x-c)**2 for c in centroids]) for x in data])
        
        # Hitung probabilitas untuk memilih setiap data point sebagai centroid berikutnya
        probabilities = distances / distances.sum()
        
        # Pilih data point baru berdasarkan probabilitas
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        next_centroid_index = np.searchsorted(cumulative_probabilities, r)
        centroids.append(data[next_centroid_index])
    
    return np.array(centroids)

# Folder yang berisi gambar
folder_path = PATH_TEST  # Ganti dengan path folder Anda

# List untuk menyimpan semua pixels dan gambar
all_images = []
all_pixels = []

# Tampilkan gambar asli
def show_image(image, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def predict_clustering(pixels, centroids, normalize=True):
    """
    Melakukan prediksi cluster untuk setiap data point (pixels) berdasarkan centroid yang sudah ada.

    Parameters:
    pixels : array-like, shape (n_samples, 3)
        Data input yang akan diklasifikasikan ke cluster (RGB values of pixels).
    centroids : array-like, shape (n_clusters, 3)
        Centroid dari setiap cluster (RGB values).
    normalize : bool, default=True
        Jika True, maka nilai RGB akan dinormalisasi menjadi rentang [0, 1].

    Returns:
    labels : array, shape (n_samples,)
        Label cluster untuk setiap pixel.
    """
    # Normalisasi nilai RGB ke [0, 1] jika diperlukan
    if normalize:
        pixels = pixels / 255.0
        centroids = centroids / 255.0

    # Hitung jarak dari setiap pixel ke setiap centroid
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
    
    # Tentukan cluster dengan jarak terdekat untuk setiap pixel
    labels = np.argmin(distances, axis=1)
    
    return labels


def visualize_clusters(image, labels, k, centroids):
    """
    Membuat gambar baru berdasarkan hasil cluster, mewarnai piksel sesuai dengan warna dominan dari centroid.

    Parameters:
    image : array-like, shape (height, width, 3)
        Gambar input asli (3 channel RGB).
    labels : array-like, shape (n_samples,)
        Label cluster untuk setiap piksel dalam gambar.
    k : int
        Jumlah cluster.
    centroids : array-like, shape (k, 3)
        Centroid (warna dominan) dari setiap cluster (dalam skala 0-255).

    Returns:
    clustered_rgb_image : array-like, shape (height, width, 3)
        Gambar berwarna sesuai dengan hasil clustering.
    """
    # Buat array kosong untuk gambar cluster
    clustered_rgb_image = np.zeros_like(image)

    # Reshape label menjadi ukuran gambar (2D) untuk akses piksel yang lebih mudah
    height, width, _ = image.shape
    labels_reshaped = labels.reshape(height, width)

    # Assign warna dominan ke piksel sesuai label clusternya
    for cluster in range(k):
        # Ambil warna centroid untuk cluster ini
        dominant_color = centroids[cluster]
        # Warnai semua piksel yang masuk ke dalam cluster ini
        clustered_rgb_image[labels_reshaped == cluster] = dominant_color

    return clustered_rgb_image


# Fungsi untuk menampilkan gambar dengan legenda
import streamlit as st

def show_image_with_legend(image, centroids, k, title):
    plt.figure(figsize=(8, 8))  # Ukuran gambar lebih besar untuk memberi ruang pada legenda
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

    # Buat list untuk memetakan warna centroid ke legenda
    legend_labels = []
    for i in range(k):
        # Buat patch warna untuk setiap cluster
        color_patch = plt.Rectangle((0, 0), 1, 1, fc=centroids[i] / 255.0)
        legend_labels.append(color_patch)

    # Tambahkan lebih banyak ruang di bawah gambar untuk legenda
    plt.subplots_adjust(bottom=0.2)  # Memberi ruang ekstra di bawah gambar untuk legenda
    
    # Tambahkan legenda di bawah gambar
    plt.legend(legend_labels, [f'Cluster {i + 1}' for i in range(k)],
               loc="lower center", ncol=k, bbox_to_anchor=(0.5, -0.1),
               frameon=False, borderpad=1, handletextpad=1, columnspacing=1)

    # Tampilkan gambar dengan streamlit
    st.pyplot(plt)
labels = []