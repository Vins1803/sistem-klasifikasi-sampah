import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# ================================
# Fungsi set background
# ================================
def set_background(image_url):
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-attachment: fixed;
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("https://d39wptbp5at4nd.cloudfront.net/media/5889_original_600_Bank_Sampah_Ditargetkan_Terbangun_Tahun_Ini.jpg")

# ================================
# Load model dan class labels
# ================================
model = load_model('vgg16_garbage_model_v2.keras')

try:
    with open("class_labels.json", "r") as f:
        class_indices = json.load(f)
    class_labels = {v: k for k, v in class_indices.items()}
except Exception as e:
    st.error(f"Gagal memuat class_labels.json: {e}")
    st.stop()

# ================================
# Kategori Sampah
# ================================
recycle_classes = [
    'cardboard', 'paper', 'plastic', 'metal',
    'white-glass', 'green-glass', 'brown-glass'
]
non_recyclable = ['battery', 'trash', 'clothes', 'shoes']
organic_classes = ['biological']

nama_material = {
    'plastic': 'Plastik',
    'metal': 'Kaleng / Logam',
    'paper': 'Kertas',
    'cardboard': 'Kardus',
    'green-glass': 'Botol Kaca Hijau',
    'brown-glass': 'Botol Kaca Coklat',
    'white-glass': 'Botol Kaca Putih',
    'battery': 'Baterai',
    'trash': 'Sampah Campuran',
    'clothes': 'Pakaian Bekas',
    'shoes': 'Sepatu Bekas',
    'biological': 'Daun / Sisa Makanan'
}

# ================================
# UI Streamlit
# ================================
st.title("Klasifikasi Sampah Daur Ulang")
st.write("Upload gambar sampah, dan sistem akan memprediksi jenis dan kategorinya.")

uploaded_file = st.file_uploader("Pilih gambar (maks 5 MB)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > 5:
        st.warning(f"Ukuran file {file_size_mb:.2f} MB melebihi batas maksimum 5 MB.")
    else:
        img = Image.open(uploaded_file)
        st.image(img, caption='Gambar yang Diupload', use_container_width=True)

        if st.button('Prediksi'):
            try:
                img_resized = img.resize((150, 150))
                img_array = img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                prediction = model.predict(img_array)
                predicted_index = np.argmax(prediction[0])
                label = class_labels.get(predicted_index, "unknown")
                jenis_sampah = nama_material.get(label, label.capitalize())

                if label in recycle_classes:
                    status = "Sampah ini adalah sampah yang dapat didaur ulang."
                elif label in organic_classes:
                    status = "Sampah ini adalah sampah organik (tidak dapat didaur ulang industri)."
                elif label in non_recyclable:
                    status = "Sampah ini tidak dapat didaur ulang."
                else:
                    status = "Kategori sampah tidak diketahui."

                kategori = f"Termasuk Sampah: {jenis_sampah}"

                st.markdown(f"""
                <div style="
                    background-color: rgba(255, 255, 255, 0.85);
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                    color: #000000;
                    font-size: 18px;
                ">
                <b>Jenis Sampah:</b> {jenis_sampah}<br>
                <b>{kategori}</b><br>
                {status}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
