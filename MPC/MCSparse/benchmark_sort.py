"""
Toy module to figure out efficiency of various functions.
"""

from functools import wraps
from itertools import islice
from collections import defaultdict
import time

import numpy as np 
from numba import jit
import scipy.sparse as sps

#-----------------------
def timefunc(func):
  """
  Simple timer.
  Parameters:
    func (callable) : function to time.
  """
  @wraps(func)
  def wrap(*args, **kwargs):

     ts = time.time()
     result = func(*args, **kwargs)
     te = time.time()

     td = te - ts
     print("Elapsed time {0}".format(td))

     return result
  return wrap

@timefunc
def slc_dd(iptr, idcs):
  storage = {tuple(islice(idcs, int(x))) : 1 for x in iptr}
  
  return storage

import numba 

a = sps.rand(100,100, density = 0.05, format = 'csr')

@numba.jit(nopython=True)
def assign_labels_numba(idcs, iptr, labels):
    n_row = iptr.shape[0] - 1

    i_label = 0

    for i in range(n_row):
        i_start = iptr[i]
        i_end = iptr[i+1]

        for j in range(i_start, i_end):
            labels[idcs[j]] = i_label

        i_label += 1

    return labels


#----------------------------------------------------
def generate_amat(block_sizes, keep_density = 0.1, fill_value = 1):
  """
  Creates a block diagonal sparse adjacency matrix.
  Parameters:
    order ([int]) : list of number of nodes in a block 
    keep_density (float) : what proportion of the rows are kept. Default = 0.1.
    fill_value (int) : value to fill the blocks with.
  Returns:
    (scipy.sparse.csr_matrix) : block diagonal sparse matrix
  """

  if not all(map(lambda x: isinstance(x, int), block_sizes)):
    raise TypeError("orders must be of type int")
  if not all(block_sizes):
    raise ValueError("orders must be positive")
  if not (keep_density > 0.0) or (keep_density > 1.0):
    raise ValueError("0.0 < keep_density <= 1.0")

# --- calculate total size
  n_row = sum(block_sizes)

# --- create empty csr matrix
  adj_mat = sps.csr_matrix((n_row, n_row), dtype = np.int)

# --- number of selected rows
  n_row_keep = np.int(n_row * keep_density)

# select rows to keep
  row_idcs_keep = np.random.choice(n_row, size = n_row_keep, replace = False)
  row_idcs_keep.sort() # make sure write happens sequentially

# calculate indptr
  block_sizes_ = np.array(block_sizes)
  num_elements_in_rows = np.repeat(block_sizes_, block_sizes_)
  mask = np.full_like(num_elements_in_rows, 0)
  mask[row_idcs_keep] = 1
  indptr = np.cumsum(num_elements_in_rows * mask)

# calculate column indices
  offset = np.cumsum(np.insert(block_sizes_[:-1], 0, 0))
  offset = np.repeat(offset, block_sizes_)[row_idcs_keep]

  col_idx_ranges = num_elements_in_rows[row_idcs_keep]
  indices = np.concatenate([np.add(np.arange(x), y) for x, y in zip(col_idx_ranges, offset)])

# pass row pointers
  adj_mat.indptr[1:] = indptr
# pass column indices
  adj_mat.indices = indices
# create data
  adj_mat.data = np.full_like(adj_mat.indices, fill_value)

  return adj_mat

# ----------------------------------
def csr_full_matrix_factory(shapes, fill_value):
  """
  Generates a sequences of csr sparse matrices. The matrices are full, but in sparse format, 
  so that they can easily be processed by sparse matrix constructors.
  Parameters:
    shapes (sequence of tuples) : the sequence of shapes
    fill_value (int) : all matrix elements will have this value
  Returns:
    (()) : generator object of length shapes. It generates csr matrices
  """
# --- iterate through shapes
  for nrow, ncol in shapes:
# create uniform data of the correct number
    data = np.full((nrow * ncol), fill_value = 1, dtype = np.int)
# column indices
    indices = np.tile(np.range(ncol), nrow)
# number of nonzero elements in the rows
    indptr = np.arange(nrow + 1, dtype = np.int) * ncol
# create matrix
    yield sps.csr_matrix(data, indices, indptr)

# ----------------------------------
def create_block_diagonal_matrix(block_sizes, fill_value = 1, keep_density = 0.01,  keep_each_block = True):
  """
  Creates a block diagonal csr matrix.
  Parameters:
    block_sizes ([int]) : sizes of the blocks.
    keep_density (float) : the proportion of rows to be kept. Default 0.01.
    fill_value (int) : the value of the elements. Default 1.
    keep_each_block (bool) : whether to keep at least one row from each block. Default True
  """

  block_sizes_ = np.array(block_sizes, dtype = np.int)
  n_keep_rows = np.rint(block_sizes_ * keep_density).astype(np.int)

# keep one row from each block all blocks
  n_keep_rows[n_keep_rows == 0] = 1
# create shapes
  shapes = (n_keep_rows, block_sizes_)
# set up generator for the sequence of matrices
  matrices = csr_full_matrix_factory(shapes, fill_value)

# create a blockdiagonal matrix by concatenating the blocks
  adj_mat = sps.block_diag(matrices, format = 'csr', dtype = np.int)

  return adj_mat

block_sizes = list(range(100, 200, 1))

import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()
a = create_block_diagonal_matrix(block_sizes, fill_value = 1, keep_density = 0.1, keep_each_block = True)
pr.disable()

# --- print profiler results
pr.create_stats()
pr.print_stats(sort = 'time')



