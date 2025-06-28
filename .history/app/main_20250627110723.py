import os
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

# Load model dan scaler
working_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(working_dir, "trained_model", "knn_sugarcane_model_fixed.pkl"))
scaler = joblib.load(os.path.join(working_dir, "trained_model", "scaler_sugarcane.pkl"))

# Mapping label
class_labels = {
    0: 'healthy',
    1: 'redrot',
    2: 'rust',
    3: 'yellow'
}

# Fungsi ekstraksi fitur GLCM
def extract_glcm_features(image):
    gray = rgb2gray(np.array(image))
    gray = (gray * 255).astype('uint8')

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        (glcm ** 2).sum(),  # ASM
        gray.mean(),  # mean
        np.median(gray),  # median
        -np.sum(glcm * np.log2(glcm + (glcm == 0)))  # entropy
    ]
    return np.array(features).reshape(1, -1)

# Streamlit UI
st.title("Sugarcane Disease Classifier (KNN + GLCM)")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Classify"):
        try:
            features = extract_glcm_features(image)
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            st.success(f"Prediction: {class_labels[prediction]}")
        except Exception as e:
            st.error(f"Error during classification: {str(e)}")
