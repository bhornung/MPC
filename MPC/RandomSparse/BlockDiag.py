import numpy as np
import scipy.sparse as sps

from MatrixFactory import make_csr_full_matrix, CsrFullMatrixFactory

#----------------------------------------------------
def old_create_block_diagonal_csr_matrix_np(block_sizes, keep_density = 0.1, fill_value = 1):
  """
  Creates a block diagonal sparse adjacency matrix.
  Parameters:
    order ([int]) : list of number of nodes in a block 
    keep_density (float) : what proportion of the rows are kept. Default = 0.1.
    fill_value (int) : value to fill the blocks with.
  Returns:
    (scipy.sparse.csr_matrix) : block diagonal sparse matrix
  """

  if not all([isinstance(x, int) for x in block_sizes]):
    raise TypeError("orders must be of type int")
  if not all(block_sizes):
    raise ValueError("orders must be positive")
  if not (keep_density > 0.0) or (keep_density > 1.0):
    raise ValueError("0.0 < keep_density <= 1.0")

# --- calculate total size
  n_row = sum(block_sizes)
  block_sizes_ = np.array(block_sizes)

# --- create empty csr matrix
  adj_mat = sps.csr_matrix((n_row, n_row), dtype = np.int)

# --- number of selected rows
  n_row_keep = np.int(n_row * keep_density)

# make sure we keep at least one row from each block
  sf = np.insert(np.cumsum(block_sizes_[:-1]), 0, 0)

# select rows to keep
  row_idcs_keep = np.random.choice(n_row, size = n_row_keep, replace = False)
  row_idcs_keep = np.unique(np.concatenate((row_idcs_keep, sf)))
  row_idcs_keep.sort() # make sure write happens sequentially
  n_row_keep = row_idcs_keep.size

# calculate indptr
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
def create_block_diagonal_csr_matrix_sps(block_sizes, factory,
                                         fill_value = 1, 
                                         keep_density = 0.1,
                                         keep_each_block = True):
  """
  Creates a block diagonal csr matrix.
  Parameters:
    block_sizes ([int]) : sizes of the blocks.
    factory (callable) : should generate a list of csr matrices.
      signature ~([(int,int)], float) --> iter(scipy.sparse.csr_matrix)

    fill_value (int) : the value of the elements. Default 1.
    keep_density (float) : the proportion of rows to be kept. Default 0.01.
    keep_each_block (bool) : whether to keep at least one row from each block. Default True
  """
# number of rows to be kept
  block_sizes_ = np.array(block_sizes, dtype = np.int)
  n_keep_rows = np.rint(block_sizes_ * keep_density).astype(np.int)

# keep one row from each block all blocks
  if keep_each_block:
    n_keep_rows[n_keep_rows == 0] = 1
# create shapes
  shapes = list(zip(n_keep_rows, block_sizes_))
# set up generator for the sequence of matrices
  mats = factory(shapes, fill_value)

# create a blockdiagonal matrix by concatenating the blocks
  adj_mat = sps.block_diag(mats, format = 'csr')

  return adj_mat

# ----------------------------------
def make_full_csr_matrix_list(shapes, fill_value):
  """
  Generates a list of csr sparse matrices. The matrices are full, but in sparse format, 
  so that they can easily be processed by sparse matrix constructors.
  Parameters:
    shapes (sequence of tuples) : the sequence of shapes
    fill_value (int) : all matrix elements will have this value
  Returns:
    matrices [scipy.sparse.csr_matrix]: list of csr matrices
  """
  matrices = []

# --- iterate through shapes
  for shape in shapes:
    matrices.append(make_csr_full_matrix(shape, fill_value))

  return matrices


# ----------------------------------
def create_block_diagonal_csr_matrix_np(block_sizes, fill_value = 1, keep_density = 0.1, keep_each_block = True):
  """
  Creates a block diagonal csr matrix.
  Parameters:
    block_sizes ([int]) : sizes of the blocks.
    factory (callable) : should generate a list of csr matrices.
      signature ~([(int,int)], float) --> iter(scipy.sparse.csr_matrix)

    fill_value (int) : the value of the elements. Default 1.
    keep_density (float) : the proportion of rows to be kept. Default 0.01.
    keep_each_block (bool) : whether to keep at least one row from each block. Default True
  """
  block_sizes_ = np.array(block_sizes, dtype = np.int)
# number of columns in each block
  n_cols = np.array(block_sizes, dtype = np.int)
# number of rows in each block
  n_rows = np.rint(block_sizes_ * keep_density).astype(np.int)

# keep one row from each block all blocks
  if keep_each_block:
    n_rows[n_rows == 0] = 1

# discard empty blocks
  n_cols = n_cols[n_rows > 0]
  n_rows = n_rows[n_rows > 0]

# total number of rows, columns and nonzero elements
  num_tot_rows = np.sum(n_rows)
  num_tot_cols = np.sum(n_cols)
  num_tot_nnz = np.sum(n_cols * n_rows)

# create empty matrix with the right dimensions
  mat = sps.csr_matrix((num_tot_rows, num_tot_cols), dtype = np.int)

# set cumulative number of nnz elements for each row
  mat.indptr[1:] = np.cumsum(np.repeat(n_cols, n_rows))

# set column indices in each row
  mat.indices = np.zeros(num_tot_nnz, dtype = np.int)
  offset = 0 # offset of column indices for each block
  ilow = 0  # position of start index in indices for each row

# fill in column indices for each row successively
  for ncol, nrow in zip(n_cols, n_rows):
    ihgh = ilow + nrow * ncol
    mat.indices[ilow:ihgh] = np.tile(np.arange(offset, offset + ncol), nrow)
    offset += ncol
    ilow = ihgh + 0

# set data
  mat.data = np.full(num_tot_nnz, fill_value)

  return mat

if __name__ == "__main__":
  block_sizes = [500] * 500

  from time import perf_counter
  import cProfile
  pr = cProfile.Profile()
  pr.enable()
  mat = create_block_diagonal_csr_matrix_np(block_sizes, keep_density = 0.1)
  pr.disable()

  pr.create_stats()
  pr.print_stats(sort = 'cumtime')
