import numpy


def DBSCAN(Dataset, eps, MinPts):
    # Dataset adalah data
    # eps adalah epsilon value
    # MinPts adalah minimum data dalam range eps
    # Inisialisasi label
    labels = [0]*len(Dataset)

    # ClusterID adalah id cluster saat ini.
    ClusterID = 0

    # Iterasi index dari setiap instance
    for InstanceIndex in range(0, len(Dataset)):

        # Jika label instance adalah 0 maka instance tersebut belum menjadi bagian dari sebuah cluster
        # Jika label = 0 maka dapat menjadi kandidat cluster
        if not (labels[InstanceIndex] == 0):
            continue

        # Cari semua neighbor dari instance
        NeighborInstance = findNeighbor(Dataset, InstanceIndex, eps)

        # Jika jumlah neighbor dari instance kurang dari MinPts maka instance tersebut adalah noise
        if len(NeighborInstance) < MinPts:
            labels[InstanceIndex] = -1
        # Jika terdapat neighbor minimal sejumlah MinPts, maka akan dijalankan fungsi grow cluster
        else:
            ClusterID += 1
            growCluster(Dataset, labels, InstanceIndex,
                        NeighborInstance, ClusterID, eps, MinPts)

    return labels


def growCluster(Dataset, labels, InstanceIndex, NeighborInstance, ClusterID, eps, MinPts):
    # Fungsi ini bertujuan untuk menumbuhkan cluster dengan cara melakukan
    # pencarian terhadap NeighborInstance dan mencari seluruh kandidat
    # cluster di dalamnya dan menambahkan seluruh tetangganya kedalam NeighborInstance

    labels[InstanceIndex] = ClusterID

    i = 0
    # iterasi seluruh instance tetangga
    while i < len(NeighborInstance):

        Instance = NeighborInstance[i]

        # Jika instance merupakan noise, maka tidak perlu melakukan penmcarian terhadap tetangganya
        # Langsung jadikan anggota cluster
        if labels[Instance] == -1:
            labels[Instance] = ClusterID

        # Jika instance belum di klaim oleh cluster lain, maka jadikan anggota cluster
        elif labels[Instance] == 0:
            labels[Instance] = ClusterID

            # Cari seluruh tetangga dari instance
            InstanceNeighbor = findNeighbor(Dataset, Instance, eps)

            # Jika instance memiliki tetangga dengan jumlah lebih dari MinPts
            # maka tambahkan seluruh tetangganya kedalam queue neighbor untuk dilakukan pencarian
            if len(InstanceNeighbor) >= MinPts:
                NeighborInstance = NeighborInstance + InstanceNeighbor
            # Jika instance tidak memiliki jumlah tetangga yang lebih dari MinPts
            # maka instance tersebut merupakan leaf point dan tidak perlu dilakukan pencarian apda tetangganya
        i += 1


def findNeighbor(Dataset, Point, Eps):
    neighbors = []

    # Iterasi setiap instance dalam dataset
    for Pn in range(0, len(Dataset)):

        # Hitung jarak dan jika kurang dari eps maka jadikan neighbor
        if numpy.linalg.norm(Dataset[Point] - Dataset[Pn]) < Eps:
            neighbors.append(Pn)

    return neighbors


def main():

    # LOAD IRIS
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    labeldbscans = DBSCAN(X, 0.53, 14)
    print(labeldbscans)


if __name__ == "__main__":
    main()
