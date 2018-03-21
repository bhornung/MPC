"""
Module to test the numpy implementation of Markov cluster algorithm.
"""

import unittest
import numpy as np

from RandomGraph import create_disjoint_complete_graphs, get_adjacency_matrix

from MC_numpy import _add_diagonal
from MC_numpy import _row_normalise
from MC_numpy import _collect_clusters

#--------------------------------
class Test_np_add_diag(unittest.TestCase):
    """
    Tests for numpy matrix operations.
    """
#----------
    def test_add_diagonal_default(self):
        """
        Tests whether a a values has been correctly added to the diagonal. 
        """
# trial scale value and matrix
        n_row = 10
        a = np.zeros((n_row, n_row), dtype = np.float)

# add unit diagonal
        at = _add_diagonal(a)

# reference vector of two-s of length diagonal
        ref_vec = np.full((n_row), 1.0, dtype = np.float) 

# does the diagonal have the right values?
        np.testing.assert_almost_equal(np.diagonal(at), ref_vec)

# anythin else has been added outside of the diagonal
        mask = np.eye(n_row, dtype = np.bool)
        np.testing.assert_equal(at[~mask], a[~mask])

#----------
    def test_add_diagonal_with_value(self):
        """
        Tests whether a a values has been correctly added to the diagonal. 
        """
# trial scale value and matrix
        scale = 2.0
        n_row = 10
        a = np.zeros((n_row, n_row), dtype = np.float)

# add diag of two-s to matrix
        at = _add_diagonal(a, scale = scale)

# reference vector of two-s of length diagonal
        ref_vec = np.full((n_row), scale, dtype = np.float) 

# does the diagonal have the right values?
        np.testing.assert_almost_equal(np.diagonal(at), ref_vec)

# anythin else has been added outside of the diagonal
        mask = np.eye(n_row, dtype = np.bool)
        np.testing.assert_equal(at[~mask], a[~mask])

#--------------------------------
class Test_np_row_normalise(unittest.TestCase):
  """
  Tests for _row_normalise from MC_numpy.
  """

#----------------
  def test_row_normalise_nonzero_rows(self):
    """
    Tests whether normalises rows to unit. No row sum is equal to zero.
    """

# matrix with no zero entries
    n_rows = 10
    a = np.random.rand(n_rows, n_rows) + 1.0

# normalise rows to unit
    at = _row_normalise(a)
    
# one-full vector for comparison
    ref_vec = np.full((n_rows), 1.0)

# sum must be equal to one
    np.testing.assert_array_almost_equal(at.sum(axis = 1), ref_vec)

#----------------
  def test_row_normalise_zero_rows(self):
    """
    Tests whether normalises rows to unit. No row sum is equal to zero.
    """

# matrix with no zero entries
    n_rows = 10
    a = np.random.rand(n_rows, n_rows) + 1.0

# choose n rows randomly to nullify
    n_zero_rows = np.random.randint(1, high = n_rows, size = 1)
    zero_idcs = np.random.choice(np.arange(n_rows), size = n_zero_rows)
    a[zero_idcs] = 0.0

# normalise rows to unit
    at = _row_normalise(a)
    
# one-full vector for comparison
    ref_vec = np.full((n_rows), 1.0)
    ref_vec[zero_idcs] = 0.0

# sum must be equal to one
    np.testing.assert_array_almost_equal(at.sum(axis = 1), ref_vec)

#--------------------------------
class Test_collect_clusters_one_cluster(unittest.TestCase):
    """
    Test for _collect_clusters. One cluster.
    """
# --------
    def setUp(self):
      """
      Operate on a matrix of identical size.
      """
      
      self.n_rows = 10
      self.node_idcs = np.arange(self.n_rows, dtype = np.int)
      self.cluster_idcs = np.full(self.n_rows, 1.0)
      self.a = np.zeros((self.n_rows, self.n_rows), dtype = np.int)

# --------
    def tearDown(self):
      """
      Boooo.
      """
      del self.n_rows; del self.node_idcs; del self.cluster_idcs; del self.a

# --------
    def test_zero_column(self):
      """
      No clusters.
      """
      #self.assertRaises(ValueError, _collect_clusters(self.a))

## --------
    def test_one_column(self):
      """
      Only one column has clusters.
      """

# --- one column
      col_idcs = np.random.randint(0, high = self.n_rows)
      self.a[:,col_idcs] = self.cluster_idcs

      clusters = _collect_clusters(self.a)

# number of clusters -- we expect one
      self.assertEqual(len(clusters), 1)

# cluster must have the label 0
      self.assertEqual(list(clusters.keys())[0], 0)

# all elements should be in cluster
      np.testing.assert_array_equal(clusters[0], self.node_idcs)

# --------
    def test_multiple_columns(self):
      """
      Two or more columns with the same cluster
      """
      col_idcs = np.random.choice(self.node_idcs, size = np.random.randint(2, high = self.n_rows), replace = False)
      self.a[:,col_idcs] = self.cluster_idcs[:,None]

      clusters = _collect_clusters(self.a)

# number of clusters -- we expect one
      self.assertEqual(len(clusters), 1)

# cluster must have the label 0
      self.assertEqual(list(clusters.keys())[0], 0)

# all elements should be in cluster
      np.testing.assert_array_equal(clusters[0], self.node_idcs)


#---------------------------------
class Test_collect_clusters_multiple_clusters(unittest.TestCase):

  def test_mc(self):

# @TODO make it random
    n_rows = 9
    cvec = np.zeros((3,9), dtype = np.int)
    cvec[0,0] = cvec[0,1] = cvec[0,2] = 1 
    cvec[1,3] = cvec[1,4] = cvec[1,5] = 1 
    cvec[2,6] = cvec[2,7] = cvec[2,8] = 1 

    a = np.zeros((n_rows, n_rows), dtype = np.int)
    a[:,0] = a[:,2] = a[:,3] = cvec[0]
    a[:,1] = a[:,6] = cvec[1]
    a[:,4] = a[:,5] = a[:,7] = cvec[2]

    clusters = _collect_clusters(a)

#---------------------------------
if __name__ == '__main__':
    unittest.main()
