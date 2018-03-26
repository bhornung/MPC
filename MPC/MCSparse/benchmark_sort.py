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
def generate_amat(orders, keep_density = 0.1, fill_value = 1):
  """
  Creates a block diagonal sparse adjacency matrix.
  Parameters:
    order ([int]) : list of number of nodes in a block 
    keep_density (float) : what proportion of the rows are kept. Default = 0.1.
    fill_value (int) : value to fill the blocks with.
  Returns:
    (scipy.sparse.csr_matrix) : block diagonal sparse matrix
  """

  if not all(map(lambda x: isinstance(x, int), orders)):
    raise TypeError("orders must be of type int")
  if not all(orders):
    raise ValueError("orders must be positive")
  if not (keep_density > 0.0) or (keep_density > 1.0):
    raise ValueError("0.0 < keep_density <= 1.0")

# --- calculate total size. Take rounding errors into account
  n_row = sum(orders)

# --- create empty lil matrix
  adj_mat = sps.lil_matrix((n_row, n_row), dtype = np.int)

  n_row_cum = 0

# --- loop over blocks
  for order in orders:

# row indices in block
    row_idcs = np.arange(n_row_cum, n_row_cum + order)

# calculate the number of rows that are kept
    n_keep_rows = int(order * keep_density)
# choose them randomly
    keep_rows_idcs = np.random.choice(row_idcs, size = n_keep_rows, replace = False)
    keep_rows_idcs.sort() # make sure write happens sequentially

# fill in block
# add column indices to selected rows
    dv = [fill_value] * order
    cv = row_idcs.tolist()
    for j in keep_rows_idcs:
      adj_mat.rows[j] = cv
      adj_mat.data[j] = dv

#    adj_mat.rows[keep_rows_idcs] = [row_idcs.tolist()] * n_keep_rows
# fill with data
#    adj_mat.data[keep_rows_idcs] = [[fill_value] * order] * n_keep_rows

# increase cumulative number of rows
    n_row_cum += order

  return adj_mat #.tocsr()

orders = [10000,20000,330,333,3333,3333,121,33,888]


import cProfile, pstats, io

pr = cProfile.Profile()
pr.enable()
a = generate_amat(orders, keep_density = 0.2)
pr.disable()

# --- print profiler results
pr.create_stats()
pr.print_stats(sort = 'time')