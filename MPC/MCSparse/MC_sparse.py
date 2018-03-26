"""
Scipy sparse implementation of the Markov cluster algorithm.
"""

import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sps
import scipy.sparse.linalg as LA 

from BaseMC.BaseMC import BaseMC
#-----------------------
def _add_diagonal(X, scale):
  """
  Add scale * identity matrix to a matrix
  Parameters:
    X (scipy.sparse.csr_matrix) : 2D sparse matrix
    scale ({int,float}) : the number to be added to the diagonal
  Returns:
    None
  """

  if not isinstance(scale, (int, float)):
    raise TypeError("Scale parameter must be int or float. Got: {0}".format(type(scale)))

# --- fill diagonal with values
  X.setdiag(scale)


#-----------------------
def _collect_clusters(X):
  """
  Assigns cluster labels to nodes.
  Parameters:
    X ({scipy.sparse.csr_matrix,scipy.sparse.csc_matrix}) : transition probability matrix
  Return:
    clusters ({int:[int]}) : indices of nodes grouped according to clusters.
  """

  # Motivation: We start with a matrix that does not have zero rows/columns.
  # The identical c/r-s represent the same cluster.
  # Finding these one read out the indices of nodes from them.

# Columns of the same pattern should be identical, therefore we do not check for the data.
# Therefore, we only compare for the nonzero (nnz) indices.

  if X.format != 'csr' and X.format != 'csc':
    raise TypeError("only 'csr' and 'csc' matrices are accepted")

# --- copy nnz indices
  ind1 = X.indices

# create a coo representation by inflating the r/c pointers
  num_nnz_el = np.diff(X.indptr)
  n_rc = num_nnz_el.size
  ind2 = ([_idx] * _nel for _idx, _nel in zip(range(n_rc, dtype = np.int), num_nnz_el))
  ind2 = filter(lambda x: x != [], ind2)
  ind2 = np.array(ind2, dtype = np.int)

  (islice(ind1, n_el) for n_el in ind2)

#-----------------------
def _cull(X, threshold):
  """
  Set elements to zero if they are close to zero.
  Parameters:
    X (scipy.sparse.csr_matrix) : zee matrix
  Returns:
    None
  """
  X[X < threshold] = 0.0

# @TODO split to two functions 1) in-place multiplication, 2) test for idempotency
# so that the we save one copying. Test can be called at certain intervals.
#-----------------------
def _expand(X, expand_power):
  """
  Raises a square matrix to the expand_power-th power
  Parameters:
    X (scipy.sparse.csr_matrix) : 2D sparse matrix
    expand_power (int) : power (number) of matrix multiplications
  Returns:
    Xm scipy sparse : exponentiated matrix
    diff_ (np.float) : the sum of elementwise absolut difference between the original and exponentiated matrix
  """

  if not isinstance(expand_power, int):
    raise ValueError("expand_power must be of type int. Got: {0}".format(type(expand_power)))

  if expand_power < 1:
    raise ValueError("expand_power must be positive")

# mild danger. We are passing the reference to the original object
  if expand_power == 1:
    diff_ = 0.0
    return X, diff_

# calculate matrix power >= 2
# keep original matrix for comparison
  Xm = X.copy()

# in place multiplication
  for idx in range(1, expand_power):
    Xm *= Xm 

# deviation from being idempotent
  diff_ = abs(Xm - X).sum()

  return Xm, diff_

#----------------------- 
def _inflate(X, inflate_power):
  """
  Raises the matrix to inflate power elementwise in place.
  Parameters:
    X (scipy.sparse.csr_matrix) : 2D sparse matrix
    inflate_power : power to raise matrix to
  Returns:
    None
  """

# type check power to avoid possible broadcasting
  if not isinstance(inflate_power, (int, float)):
    raise TypeError("inflate power must be float or int. Got: {0}".format(type(inflate_power)))

# calculate elementwise power inplace
  X.data **= inflate_power

#-----------------------
def _init_matrix(X, diag_scale):
  """
  Generates a transition matrix from the connectivity matrix
  Parameters:
    X (scipy.sparse.csr) : The connectivity matrix a 2D square matrix.
    diag_scale ({int, float}) : factor to scale the diagonal with. 
  Returns:
    X (scipy.sparse.csr)
  """

# --- add self loops
  _add_diagonal(X, diag_scale)

# --- normalise outgoing probabilities to unit
  X = _row_normalise(X)

  return X

#---------------------
def _markov_cluster(X, expand_power, inflate_power, max_iter, tol, threshold):
  """
  Performs maximum max_iter number Markov cluster cycles.
  Parameters:
    X : matrix
    expand_power (int) : number of Markov process steps
    inflate_power {int, float} : the normalising parameter for the transition probabilities.
    max_iter (int) : maximum number of iterations. 
  """

# --- do maximum max_iter iterations
  for idx in np.arange(max_iter):

# perform one cycle of Markov iteration
    _inflate(X, inflate_power)
    _cull(X, threshold)
    _row_normalise(X)
    X, diff_ = _expand(X, expand_power)

# check whether convergence reached <-- matrix is idempotent
    if diff_ < tol:
      break

  return X

#-----------------------
def _row_normalise(X):
  """
  Normalises the rows of a matrix to unit
  Parameters:
    X (scipy.sparse matrix) : matrix
  Returns:
    None
  """

  row_norm = np.ravel(X.sum(axis = 1))
  nnz_n = np.take(row_norm, X.indices)
  nnz_n = np.where(nnz_n == 0.0, 1.0, nnz_n)

  X.data /= nnz_n

#------------------------
class MCsparse(BaseMC):
  """
  A scipy.sparse implementation of the Markov Cluster algorithm.
  """

#-----------------------
  def __init__(self, diag_scale = 1.0, expand_power = 2, inflate_power = 2, 
               max_iter = 10, threshold = 0.00001, tol = 0.001):

# --- invoke parent initialisation
    super().__init__(diag_scale = diag_scale, 
                     expand_power = expand_power, inflate_power = inflate_power,
                     max_iter = max_iter, threshold = threshold, tol = tol)

#------------------------
  def fit(self, X):
    """
    Clusters a graph using Markov cluster algorithm.
    Parameters:
      X (scipy.sparse.csr_matrix): 2D connectivity matrix
    """

# --- create transition probability matrix
    X_ = _init_matrix(X, self.diag_scale)

# --- perform clustering
    _markov_cluster(X_, self.expand_power, self.inflate_power, 
                    self.max_iter, self.tol, self.threshold)

# assign labels
    clusters = _collect_clusters(X_)
    self._labels_ = np.zeros(X_.shape[0], dtype = np.int)

    for _cluster_id, _cluster_members in clusters.items():
      self._labels_[_cluster_members] = _cluster_id

    return self

#------------------------
  def fit_predict(X):
    """
    Fits the graph with clusters and returns the cluster labels.
    Parameters:
      X (scipy.sparse.csr_matrix): 2D connectivity matrix
    Returns:
      self.labels_ (np.ndarray[X.shape[0]]) : index of cluster to which a node belongs to.
    """

    return self.fit(X).labels_


