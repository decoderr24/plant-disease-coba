import os
import joblib
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

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
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # GLCM Features
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    f1 = graycoprops(glcm, 'energy')[0, 0]
    f2 = graycoprops(glcm, 'homogeneity')[0, 0]
    f3 = graycoprops(glcm, 'contrast')[0, 0]
    f4 = graycoprops(glcm, 'correlation')[0, 0]

    # RGB Mean
    r = img_np[:, :, 0]
    g = img_np[:, :, 1]
    b = img_np[:, :, 2]
    f5 = np.mean(r)
    f6 = np.mean(g)
    f7 = np.mean(b)

    # RGB Std Dev
    f8 = np.std(r)
    f9 = np.std(g)

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9]).reshape(1, -1)

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
        

