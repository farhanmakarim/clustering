import numpy as np
import pandas as pd
import random


class K_Means:
    def __init__(self, k, max_iterations):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, data):

        self.centroids = {}

        # Inisialisasi centroid dengan k instance pertama
        for i in range(self.k):
            self.centroids[i] = data[i]

        # centroid_indexes = random.sample(range(0, len(data)-1), self.k)
        # for i in range(self.k):
        #     self.centroids[i] = data[centroid_indexes[i]]

        # Iterasi
        for i in range(self.max_iterations):
            self.classes = {}
            self.labels = []
            for i in range(self.k):
                self.classes[i] = []

            # Hitung jarak setiap titik ke centroid dan ambil yang paling kecil jaraknya
            for features in data:
                distances = [np.linalg.norm(
                    features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)
                self.labels.append(classification)

            # previous = dict(self.centroids)
            previous = self.centroids

            # Hitung rata-rata cluster untuk membentuk centroid baru
            for classification in self.classes:
                self.centroids[classification] = np.average(
                    self.classes[classification], axis=0)

            isOptimal = True
            # Cek apakah centroid berubah
            for index, centroid in enumerate(self.centroids):
                if np.any(previous[index] != centroid):
                    isOptimal = False

            # Jika centroid tidak berubah, hentikan iterasi
            if isOptimal:
                break

    def pred(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid])
                     for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


def main():

    # LOAD IRIS
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    km = K_Means(3, 100)
    km.fit(X)
    print(km.labels)


if __name__ == "__main__":
    main()
