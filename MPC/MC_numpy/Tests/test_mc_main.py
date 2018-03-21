import unittest

from itertools import chain, product

from uuid import uuid1
import numpy as np

from MC_numpy.MC_numpy import MCnumpy
from RandomGraph import create_disjoint_complete_graphs, get_adjacency_matrix

class Test_mc_init(unittest.TestCase):
  """
  Test initialiser.
  """

# @TODO test baseclass instead
#----------------------------------------
class Test_mc_set_params(unittest.TestCase):
  """
  Test setting parameters after initialisation.
  """
# --------
  def setUp(self):
    """
    Creates and initialises a classifier.
    """
# @TODO make a strenum outside of classes        
# --- Markov cluster init arguments. Define them explicitly for greater control.
    self.mc_kwargs = {"diag_scale" : 1.0,
                "expand_power" : 2,
                "inflate_power" : 2,
                "max_iter" : 10,
                "threshold" : 0.00001,
                "tol" : 0.001}

# --- initialise Markov cluster classifier
    self.clus = MCnumpy(**self.mc_kwargs)

# --------
  def tearDown(self):
    """
    Clean up classifier.
    """
    del self.clus
    del self.mc_kwargs

# -------
  def test_invalid_keyword_assigment(self):
    """
    Tests whether an invalid parameter name raises exception.
    """
# --- generate a parameter name that is the most likely to be invalid
    parameter_name = str(uuid1())
    parameter_val = str(uuid1())
    param_dict = {parameter_name : parameter_val}

# --- make sure it raises an error
    with self.assertRaises(KeyError):
      self.clus.set_params(param_dict)

# @TODO write separate tests for each keyword in addition to this
# ------
  def test_valid_keyword_assigment(self):
    """
    Test whether valid keywords are assigned properly.
    """
# create a new dictionary of parameters. Set values implicitly. 
    param_dict = self.mc_kwargs
    param_dict["diag_scale"] = self.mc_kwargs["diag_scale"] + 1
    param_dict["expand_power"] = self.mc_kwargs["expand_power"] + 1
    param_dict["inflate_power"] = self.mc_kwargs["inflate_power"] + 1
    param_dict["max_iter"] = self.mc_kwargs["max_iter"] + 1
    param_dict["threshold"] = self.mc_kwargs["threshold"] + 1
    param_dict["tol"] = self.mc_kwargs["tol"] + 1

# --- try to pass all parameters, so that we can check for cross assigment
# when to parameter names refer to the same backing field
    self.clus.set_params(param_dict)

# compare updated parameter values to refrence values
    for _pname in param_dict.keys():
      self.assertAlmostEqual(self.clus.get_params()[_pname], param_dict[_pname])
      
#-------------------------------------
class Test_mc_cluster(unittest.TestCase):
    """
    Sanity checks whether simple cluster structures are found.
    """
# --------
    def setUp(self):
      """
      Creates and initialises a classifier.
      """
        
# --- Markov cluster init arguments. Define them explicitly for greater control.
      self.mc_kwargs = {"diag_scale" : 1.0,
                  "expand_power" : 2,
                  "inflate_power" : 2,
                  "max_iter" : 10,
                  "threshold" : 0.00001,
                  "tol" : 0.001}

# --- initialise Markov cluster classifier
      self.clus = MCnumpy(**self.mc_kwargs)

# --------
    def tearDown(self):
      """
      Removes the classifier.
      """

      del self.mc_kwargs
      del self.clus

# --------
    def test_one_cluster(self):
      """
      Test whether one cluster is identified correctly.
      """

# --- create a complete graph with n_nodes nodes and its adjacency matrix
      n_nodes = 4
      ref_labels = np.zeros((n_nodes), dtype = np.int)
      g = create_disjoint_complete_graphs([n_nodes])
      X = get_adjacency_matrix(g, todense = True)

# --- try to fit it
      self.clus.fit(X)

# --- all labels should be zero
      np.testing.assert_array_equal(self.clus.labels_, ref_labels, err_msg = "Failed to find a single clusters.")

# --------
    def test_multiple_clusters(self):
      """
      Test whether multiple clusters are found.
      """

# create disjoint complete matrices
# @TODO make this random --> I will write a separate utility
      orders = [4,3,2,1]
      ref_labels = list(chain(*[[_lb]*_rep for _lb, _rep in zip(range(len(orders)), orders)]))
      g = create_disjoint_complete_graphs(orders)
      X = get_adjacency_matrix(g, todense = True)

# --- try to fit
      self.clus.fit(X)

# --- compare to reference labels
      np.testing.assert_array_equal(self.clus.labels_, ref_labels, err_msg = "Failed to find multiple clusters")

if __name__ == '__main__':
    unittest.main()
