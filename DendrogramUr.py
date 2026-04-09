import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# ==========================
# Pobranie wymiarów danych
# ==========================

rows = int(input("Podaj liczbę wierszy: "))
cols = int(input("Podaj liczbę kolumn: "))

# ==========================
# Pobranie etykiet
# ==========================

while True:
    labels_input = input(f"Podaj {rows} nazw wierszy (oddzielone spacją): ")
    labels = labels_input.strip().split()

    if len(labels) != rows:
        print(f"Błąd: musisz podać dokładnie {rows} nazw!")
    else:
        break

print(f"\nWprowadź dane ({rows} wierszy, każdy po {cols} wartości oddzielonych spacją):")

data = []

# ==========================
# Wczytywanie danych
# ==========================

for i in range(rows):
    while True:
        try:
            line = input(f"{labels[i]}: ")
            values = [float(x) for x in line.strip().split()]

            if len(values) != cols:
                print(f"Błąd: podaj dokładnie {cols} wartości!")
                continue

            data.append(values)
            break

        except ValueError:
            print("Błąd: upewnij się, że wszystkie wartości są liczbami.")

# Zamiana na numpy array
data = np.array(data)

# ==========================
# Obliczanie odległości
# ==========================

distance_matrix = pdist(data, metric='euclidean')

# ==========================
# UPGMA (average linkage)
# ==========================

Z = linkage(distance_matrix, method='average')

# ==========================
# Rysowanie dendrogramu
# ==========================

plt.figure(figsize=(10, 6))
dendrogram(Z, labels=labels)
plt.title("Drzewo filogenetyczne (UPGMA)")
plt.xlabel("Próbki")
plt.ylabel("Odległość euklidesowa")
plt.grid(True)
plt.show()
