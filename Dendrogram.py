import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# ==========================
# Dane
# ==========================

data = np.array([
    [-2, -2, 7, -2],
    [-0.875, -0.875, 3.625, -0.875],
    [-0.36363636, -0.36363636, 2.09090909, -0.36363636],
    [-0.07142857, -0.07142857, 1.21428571, -0.07142857],
    [0.11764706, 0.11764706, 0.64705882, 0.11764706],
    [0.25, 0.25, 0.25, 0.25],
    [0.34782609, 0.34782609, -0.04347826, 0.34782609],
    [0.42307692, 0.42307692, -0.26923077, 0.42307692],
    [0.48275862, 0.48275862, -0.44827586, 0.48275862],
    [0.53125, 0.53125, -0.59375, 0.53125],
    [0.57142857, 0.57142857, -0.71428571, 0.57142857],
])

labels = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5",
          "0.6", "0.7", "0.8", "0.9", "1.0"]

# ==========================
# Obliczanie odległości
# ==========================

distance_matrix = pdist(data, metric='euclidean')

# ==========================
# UPGMA (average linkage)
# ==========================

Z = linkage(distance_matrix, method='average')

# ==========================
# Rysowanie drzewa
# ==========================

plt.figure(figsize=(10, 6))
dendrogram(Z, labels=labels)
plt.title("Drzewo filogenetyczne (UPGMA)")
plt.xlabel("Próbki")
plt.ylabel("Odległość euklidesowa")
plt.grid(True)
plt.show()
