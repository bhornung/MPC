"""
Scipy sparse implementation of the Markov cluster algorithm.
"""

from scipy.sparse import csr_matrix
import scipy.sparse as sp


#-----------------------
def _add_diagonal(X, scale = 1.0):
  """
  Add scale * identity matrix to a matrix
  Parameters:
    X (scipy.sparse.csr_matrix) : 2D sparse matrix
    scale ({int,float}) : the number to be added to the diagonal
  Returns:
    X (scipy.sparse.crs_matrix) : matrix with modified diagonal 
  """

# create diagonal sparse matrix
  diag = sp.eye(X.get_shape()[0], format = 'csr')
  diag *= scale

# add to original matrix
  X = X + diag 

  return X

#-----------------------
def _expand(X, expand_power):
  """
  Raises a square matrix to the expand_power-th power
  Parameters:
    X (scipy.sparse.csr_matrix) : 2D sparse matrix
    expand_power ({int,float}) : power (number) of matrix multiplications
  Returns:
    X^{expand_power}
  """

# calculate matrix power in place
  X.data **= expand_power

  return X

#----------------------- 
def _inflate(X, inflate_power):
  """
  Raises the matrix to inflate power elementwise
  Parameters:
    X (scipy.sparse.csr_matrix) : 2D sparse matrix
    inflate_power : power to raise matrix to
  Returns:
    X (np.ndarray) : the elementwise exponentiated matrix
  """

# type check power to avoid possible broadcasting
  if not isinstance(inflate_power, (int, float)):
    raise TypeError("inflate power must be float or int. Got: {0}".format(type(inflate_power)))

# calculate elementwise power
  X = np.power(X, inflate_power)

  return X

#-----------------------
def _row_normalise(X):
  """
  Normalises the rows of a matrix to unit
  Parameters:
    X : matrix
  Returns:
    X : rowwise normalised matrix
  """
  _row_sum = X.sum(axis = 1)

# do not divide by zeros
  mask = _row_sum != 0 
  X[mask,:] = X[mask] / _row_sum[mask, None]

  return X

from scipy.sparse import csr_matrix

row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([2, 2, 2, 2, 2, 2])
a = csr_matrix((data, (row, col)), shape=(3, 3))
def mp(x):
  x.data **= 2
  return x
m = mp(a)
print(a)
print(m)