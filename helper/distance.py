import numpy as np
import unittest

def manhattan(a, b):
    d = 0
    for i in range(len(a)):
        d += abs(a[i] - b[i])
    return d

def euclidean(a, b):
    dsquare = 0
    for i in range(len(a)):
        dsquare += np.power(a[i] - b[i], 2)
        d = np.sqrt(dsquare)
    return d

class test(unittest.TestCase):
    def test_manhattan(self):
        self.assertEqual(manhattan([0, 0], [1, 2]), 3)
        
    def test_euclidean(self):
        self.assertEqual(euclidean([0, 0], [3, 4]), 5)
