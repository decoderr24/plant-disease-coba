import os
import joblib
import json
from PIL import Image
import numpy as np
import streamlit as st

# Path direktori kerja
working_dir = os.path.dirname(os.path.abspath(__file__))
# Load model dan class index
model = joblib.load(os.path.join(working_dir, "trained-model", "knn_sugarcane_model.pkl"))

class_indices = {
    "0": "healthy",
    "1": "redrot",
    "2": "rust",
    "3": "yellow"
}

# Fungsi ekstraksi fitur GLCM dari citra grayscale
from skimage.feature import graycomatrix, graycoprops
import cv2

def extract_features(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]
    return np.array(features).reshape(1, -1)

# Fungsi prediksi
def predict_image_class(model, image, class_indices):
    features = extract_features(image)
    predicted_class_index = model.predict(features)[0]
    return class_indices[str(predicted_class_index)]

# Streamlit UI
st.title("Klasifikasi Penyakit Daun Tebu (KNN)")

uploaded_image = st.file_uploader("Upload gambar daun tebu...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    img = Image.open(uploaded_image).resize((224, 224))

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Gambar yang Diupload", use_column_width=True)

    with col2:
        if st.button("Klasifikasikan"):
            prediction = predict_image_class(model, img, class_indices)
            st.success(f"Hasil Prediksi: {prediction}")
