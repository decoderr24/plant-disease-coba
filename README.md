# Sugarcane Disease Classifier
![localhost_8502_ (1)](https://github.com/user-attachments/assets/a055aa8f-d1e2-4a32-b252-71c5395102b6)

Aplikasi web untuk klasifikasi penyakit tebu menggunakan KNN dengan ekstraksi fitur GLCM dan RGB.

## Quick Start

### 1. Clone & Setup
```bash
git clone <repository-url>
cd plant-disease-coba/app
```
### 2. Setup Enviroment
```bash
# Membuat virtual environment
python -m venv .venv

# Aktivasi virtual environment
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```
### 3. Install Dependencies
```bash
a. cd app
pip install -r requirements.txt (Wajib)
b. pip install scikit-image opencv-python joblib pillow (jika dibutuhkan)
pip install streamlit numpy pillow opencv-python scikit-image joblib (jika dibutuhkan)
```

### 4. Download Model
Download model dari: https://drive.google.com/file/d/19zLe-qTX-xWqaQbc3WmL9M9z-GJdzALx/view?usp=sharing

Letakkan file `knn_sugarcane_model.pkl` di folder `trained_model/`

### 5. Run Aplikasi
```bash
cd app
streamlit run main.py
```
### 6. Struktur folder 
app/
├── main.py
├── requirements.txt
├── config.toml
├── credentials.toml
├── class_indices.json
├── Dockerfile
└── trained_model/
    ├── knn_sugarcane_model.pkl
    └── trained_model_link.txt


Buka browser di `http://localhost:8501`

## Cara Pakai
1. Upload gambar daun tebu (PNG/JPG/JPEG)
2. Klik "Classify"
3. Lihat hasil prediksi: healthy, redrot, rust, atau yellow

## Alternatif: Docker
```bash
cd app
docker build -t sugarcane-classifier .
docker run -p 80:80 sugarcane-classifier
```

## Dataset
Noval Sofyan : https://www.kaggle.com/datasets/novalsofyan/dtm1kv1/data
