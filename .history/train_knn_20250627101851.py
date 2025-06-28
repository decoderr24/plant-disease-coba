import joblib
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Contoh: load data fitur dan label
# X_train = np.load('X_train.npy')
# y_train = np.load('y_train.npy')

# Ganti bagian ini dengan data asli Anda
X_train = ... # array fitur hasil ekstraksi GLCM
y_train = ... # array label (0, 1, 2, 3)

# Latih model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Simpan model
joblib.dump(knn, 'app/trained_model/knn_sugarcane_model.pkl')
print("Model KNN berhasil disimpan ulang.")