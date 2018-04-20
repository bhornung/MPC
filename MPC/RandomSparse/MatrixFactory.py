"""
Down the rabbit hole.
"""

import numpy as np
import scipy.sparse as sps

#--------------------------
def make_csr_full_matrix(shape, fill_value):
  """
  Creates a full csr matrix.
  Parameters:
    shape ((int,int)) : shape of the matrix
    fill_value (int) : matrix elements are set ot this value
  Returns:
    (scipy.sparse.csr_matrix()) : sparse matrix filled with the specified value
  """

# pass shape parameters
  nrow, ncol = shape
# create uniform data of the correct number
  data = np.full((nrow * ncol), fill_value = 1, dtype = np.int)
# column indices
  indices = np.tile(np.arange(ncol), nrow)
# number of nonzero elements in the rows
  indptr = np.arange(nrow + 1, dtype = np.int) * ncol
# create matrix
  return sps.csr_matrix((data, indices, indptr))


# ----------------------------------
class CsrFullMatrixFactory(object):
  """
  Creates an instance of a generator object which generates a sequence of csr matrices.
  The matrices are full, but in sparse format, hence they can easily be processed by sparse matrix constructors.
  Attributes:
    shapes ([(,)]) : list of tuples containing the shapes of the matrices
    fill_value (int) : each element will be set to this value
  """
# -------
  def __init__(self, shapes, fill_value):
    """
    shapes ([(,)]) : list of tuples containing the shapes of the matrices
    fill_value (int) : each element will be set to this value
    """
# set shapes of matrices
    try:
      _ = len(shapes)
    except Exception as err:
      print("parameter 'shapes' must implement the '__len__()' method")
      raise err

    self._shapes = shapes

# set value for elements
    self._fill_value = fill_value

# set counter to zero
    self.__ispent = 0

  @property
  def fill_value(self):
    return self._fill_value

  @property
  def shapes(self):
    return self._shapes

  @property
  def ispent(self):
    return self.__ispent

# -------
  def __iter__(self):
    """
    Returns a sparse matrix
    """
    while self.ispent < len(self):
    
      _shape = self.shapes[self.ispent]
      yield make_csr_full_matrix(_shape, self.fill_value)
      self.__ispent += 1

# -------
  def __len__(self):
    """
    Define a len method, so that the number of matrices can be known in advance.
    """
    return len(self.shapes)
