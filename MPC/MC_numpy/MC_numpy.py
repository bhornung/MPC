"""
A native numpy implementation of the Markov process classifier
"""

import numpy as np

# @TODO sort out bloody VS referencing.
from BaseMC.BaseMC import BaseMC

#-----------------------
def _add_diagonal(X, scale = 1.0):
  """
  Add scale * identity matrix to a matrix
  Parameters:
    X (np.ndarray[n,n]) : 2D square input matrix
    scale ({int,float}) : the number to be added to the diagonal
  Returns:
    X (np.ndarray) : matrix with modified diagonal 
  """

  X = X + np.eye(X.shape[0]) * scale

  return X

#---------------------
def _calc_nnz_L1_column_variance(X):
  """
  Calculates the L1 row variance of matrix of the nonzero elements 
  where the mean is calculated columnwise.
  \mu_(j) = \sum_{i} X_{i,j}  / \sum_{i} P(X_{i,j} != 0)
  Parameters:
    X (np.ndarray) : 2D square matrix
  Returns:
    nnz_L1_var : the average column wise L1 mean of the nonzero elements. 

  """
# 1) --- calculate column means
# sum all elements in matrix column wise (zeros too)
  col_nnz_sum = np.sum(X, axis = 0)
# count nonzero elements columnwise
  col_nnz_cnt = np.sum(X != 0, axis = 0)

# check for zero matrix --> this is not supposed to happen
  if np.sum(col_nnz_cnt) == 0:
    raise ValueError("Error: zero matrix")

# calculate columnwise mean -- do not divide by zero
  col_nnz_mean = np.where(col_nnz_cnt == 0, 0, col_nnz_cnt/col_nnz_sum)

# 2) --- calculate L1 variance
# substract mean from nonzero elements and take sum of modulus
  sum_abs_diff = np.sum(np.where(X == 0, 0, np.abs(X - col_nnz_mean[None,:])))
# normalise
  nnz_L1_var = sum_abs_diff / np.sum(X != 0)

  return nnz_L1_var

#---------------------
def _collect_clusters(X):
  """
  Collects those column indices together which belong to the same cluster
  Parameters:
    X np.ndarray
  """

# transpose matrix for it is easier to loop over rows
  Xt = X.T

#  --- Read this!
# each row is the indices of nodes that belong to the same cluster.
# As a consequence is two rows are not equal they represent to different clusters.

# --- select nonzero rows <-- these are the clusters
  keep_mask = np.sum(Xt, axis = 1) != 0
  nnz_idcs = np.arange(Xt.shape[0])[keep_mask]
  
# --- cluster groups will store the column index of the of the cluster vectors
  cluster_groups = [nnz_idcs[0]]

# --- double loop to compare rows
# @TODO use vector hashing instread
  for idx in nnz_idcs[1:]:
    is_found = False
    for jdx in len(cluster_groups):

# compare to know cluster nodes
# @TODO replace with allclose once I updated numpy to 1.14
      if all(Xt[idx] == Xt[cluster_groups[jdx]]):
        is_found = True
        break
# append as new cluster if node pattern is not found
    if not is_found:
      cluster_groups.append(idx)

# --- collect indices
  member_idcs = np.arange(Xt.shape[0])
  clusters = {idx : member_idcs[X[cluster_groups[idx]]] 
                for idx, cluster_group in enumerate(cluster_groups)}

  return clusters

#-----------------------
def _expand(X, expand_power):
  """
  Raises a square matrix to the expand_power-th power
  Parameters:
    X (np.ndarray) : 2D square matrix
    expand_power ({int,float}) : power (number) of matrix multiplications
  Returns:
    X^{expand_power}
  """

# calculate matrix power
  X = np.linalg.matrix_power(X, expand_power)

  return X

#----------------------- 
def _inflate(X, inflate_power):
  """
  Raises the matrix to inflate power elementwise
  Parameters:
    X (np.ndarray) : matrix
    inflate_power : power to raise matrix to
  Returns:
    X (np.ndarray) : the elementwise exponentiated matrix
  """

# type check power to avoid possible broadcasting
  if not isinstance(inflate_power, (int, float)):
    raise TypeError("inflate power must be float or int. Got: {0}".format(type(inflate_power)))

# calculate elementwise power in place
  X.data **= inflate_power

  return X

#---------------------
def _init_matrix(X, diag_scale):
  """
  Generates a transition matrix from a connectivity matrix
  Parameters:
    X (np.ndarray[n_col, n_col] : The connectivity matrix a 2D square matrix.
    diag_scale ({int, float}) : factor to scale the diagonal with. Default 1.0
  """

# --- add self loops
  X = _add_diagonal(X, diag_scale)

# --- normalise to unit probability
  X = _row_normalise(X)

  return X

#---------------------
def _markov_cluster(X, expand_power, inflate_power, max_iter, tol):
  """
  Performs maximum max_iter number Markov cluster cycles.
  Parameters:
    X : matrix
    expand_power (int) : number of Markov process steps
    inflate_power {int, float} : the normalising parameter for the transition probabilities.
    max_iter (int) : maximum number of iterations. 
  """

  for idx in np.arange(max_iter):

# perform one cycle of Markov iteration
    X = _inflate(X, inflate_power)
    X = _row_normalise(X)
    X = _expand(X, expand_power)

# check whether the attractors a converged
    avg_col_var = _calc_nnz_L1_column_variance(X)
    if avg_col_var < tol:
      break

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
  keep_mask = _row_sum != 0 
  X[keep_mask,:] = X[keep_mask,:] / _row_sum[keep_mask, None]

  return X


class MCnumpy(BaseMC):
  """
  Native numpy implementation of the Markov Cluster algorithm.
  """

#-----------------------
  def __init__(self, *args, **kwargs):

# --- fall back to parent implementation
    self.__doc__ = super().__doc__ # sorry I am lazy
    super().__init__(*args, **kwargs)


# to do fill in rest ...