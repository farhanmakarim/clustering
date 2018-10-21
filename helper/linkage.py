import numpy as np
import unittest

from .distance import euclidean

def single_linkage(clust_a, clust_b, points):
    d = np.inf

    for a in clust_a:
        for b in clust_b:
            d_temp = euclidean(points[a], points[b])
            d = min(d, d_temp)

    return d

def complete_linkage(clust_a, clust_b, points):
    d = np.inf * (-1)

    for a in clust_a:
        for b in clust_b:
            d_temp = euclidean(points[a], points[b])
            d = max(d, d_temp)

    return d

def average_group_linkage(clust_a, clust_b, points):
    sum_a = 0
    for i in clust_a:
        sum_a += points[i]
    centroid_a = sum_a / len(a)

    sum_b = 0
    for i in clust_b:
        sum_b += points[i]
    centroid_b = sum_b / len(b)

    return euclidean(centroid_a, centroid_b)

def average_linkage(clust_a, clust_b, points):
    sum = 0
    for i in clust_a:
        for j in clust_b:
            sum += euclidean(points[a], points[b])

    return sum / (len(clust_a) * len(clust_b))