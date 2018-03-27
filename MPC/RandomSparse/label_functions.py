"""
Toy module to figure out efficiency of various functions.
"""

from functools import wraps
from itertools import islice
from collections import defaultdict
import time

import numpy as np 
from numba import jit

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

if __name__ == "__main__":

  import cProfile, pstats, io
  pr = cProfile.Profile()
  pr.enable()
  pass
  pr.disable()

  # --- print profiler results
  pr.create_stats()
  pr.print_stats(sort = 'time')



