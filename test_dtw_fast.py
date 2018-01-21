'''
Reg test for the dtw_fast library.
'''

import unittest
import numpy as np
from numpy import inf
from scipy.spatial.distance import cdist
from dtw_fast import dtw_fast

class TestDtwFast(unittest.TestCase):
    
    def setUp(self):
        
        self.features1 = np.array([[1,2,3,4,5]]).T
        self.features2 = np.array([[1,1,2,2,2,3,4,5]]).T
        self.C1 = cdist(self.features1, self.features2)
        self.weights1 = np.array([1.0, 1.0, 1.0])
        self.large_int = 100000
        
    def test_distance(self):
        
        C1 = np.array(
            [[ 0.,  0.,  1.,  1.,  1.,  2.,  3.,  4.],
             [ 1.,  1.,  0.,  0.,  0.,  1.,  2.,  3.],
             [ 2.,  2.,  1.,  1.,  1.,  0.,  1.,  2.],
             [ 3.,  3.,  2.,  2.,  2.,  1.,  0.,  1.],
             [ 4.,  4.,  3.,  3.,  3.,  2.,  1.,  0.]])
        
        self.assertTrue(np.all(self.C1 == C1))


    def test_no_subseq_no_max_consecutive_step(self):
        
        dtw1 = dtw_fast(self.C1, self.weights1, 0, self.large_int)
        
        res1 = np.array(
            [[  0.,   0.,   1.,   2.,   3.,   5.,   8.,  12.],
             [  1.,   1.,   0.,   0.,   0.,   1.,   3.,   6.],
             [  3.,   3.,   1.,   1.,   1.,   0.,   1.,   3.],
             [  6.,   6.,   3.,   3.,   3.,   1.,   0.,   1.],
             [ 10.,  10.,   6.,   6.,   6.,   3.,   1.,   0.]])
        
        self.assertTrue(np.all(dtw1 == res1))
        
    def test_with_subseq_no_max_consecutive_step(self):
        
        dtw1 = dtw_fast(self.C1, self.weights1, 1, self.large_int)
        
        res1 = np.array(
           [[  0.,   0.,   1.,   2.,   3.,   5.,   8.,  12.],
            [  1.,   1.,   0.,   0.,   0.,   1.,   3.,   6.],
            [  2.,   3.,   1.,   1.,   1.,   0.,   1.,   3.],
            [  3.,   5.,   3.,   3.,   3.,   1.,   0.,   1.],
            [  4.,   7.,   6.,   6.,   6.,   3.,   1.,   0.]])

        
        self.assertTrue(np.all(dtw1 == res1))
        
    def test_no_subseq_with_max_consecutive_step(self):
        
        dtw1 = dtw_fast(self.C1, self.weights1, 0, 2)
        
        res1 = np.array(
            [[  0.,   0.,  inf,  inf,  inf,  inf,  inf,  inf],
             [  1.,   1.,   0.,   0.,   0.,  inf,  inf,  inf],
             [ inf,   3.,   1.,   1.,   1.,   0.,   1.,   3.],
             [ inf,   6.,   3.,   3.,   3.,   1.,   0.,   1.],
             [ inf,  10.,   9.,   6.,   6.,   3.,   1.,   0.]])

        
        self.assertTrue(np.all(dtw1 == res1))
        
    def test_with_subseq_with_max_consecutive_step(self):
        
        dtw1 = dtw_fast(self.C1, self.weights1, 1, 2)
        
        res1 = np.array(
            [[  0.,   0.,  inf,  inf,  inf,  inf,  inf,  inf],
             [  1.,   1.,   0.,   0.,   0.,  inf,  inf,  inf],
             [  2.,   3.,   1.,   1.,   1.,   0.,   1.,   3.],
             [  3.,   5.,   3.,   3.,   3.,   1.,   0.,   1.],
             [  4.,   7.,   8.,   6.,   6.,   3.,   1.,   0.]])

        
        self.assertTrue(np.all(dtw1 == res1))
        

if __name__ == '__main__':
    unittest.main()
