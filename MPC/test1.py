import unittest

import numpy as np

from MCSparse.MC_sparse import _row_normalise

class MPCUtilRowTest(unittest.TestCase):

#---------------
    def test_row_normalise(self, A):
        """
        Checks whether the matrix is properly normalised.
        """
# normalise matrix and get row sums
        _row_normalise(A)
        row_sum = A.sum(axis = 1)
# unit array of length number of rows
        reference_array = np.full(A.shape[0], 1, dtype = np.float)
# compare
        are_close = np.allclose(reference_array, row_sum)
        mcp_sparse_matutils.assertTrue(are_close, "Rows are not normalised correctly")

#---------------
    def test_cull(A, threshold):
        """
        Tests whether elements are removed properly.
        """
if __name__ == '__main__':
    unittest.main()

