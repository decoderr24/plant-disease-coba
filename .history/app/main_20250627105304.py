import os
import joblib
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from skimage.feature import graycomatrix, graycoprops

# Path model
working_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(working_dir, "trained_model", "knn_sugarcane_model.pkl"))

# Class label
class_labels = {
    0: "healthy",
    1: "redrot",
    2: "rust",
    3: "yellow"
}

# Fungsi ekstraksi fitur GLCM
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

# Streamlit UI
st.title('KNN Sugarcane Disease Classifier')

uploaded_file = st.file_uploader("Upload image...", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Classify"):
        features = extract_features(image)
        prediction = model.predict(features)[0]
        class_name = class_labels[prediction]
        st.success(f"Prediction: {class_name}")
