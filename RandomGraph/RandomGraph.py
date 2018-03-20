import numpy as np
import networkx as nx

#------------------------------------------
def get_adjacency_matrix(g, todense = False):
  """
  Returns the adjacency matrix of a graph.
  Parameters:
    g (nx.Graph) : a graph
    todense (bool) : If true creates a dense matrix (ordinary numpy array). False : returns a scipy.sparse matrix. Default False.
  Returns:
    a : adjacency matrix

  """
  if todense:
    return  nx.adjacency_matrix(g).todense()
  else:
    return nx.adjacency_matrix(g)

#------------------------------------------
def create_disjoint_complete_graphs(orders): 
  """
  Creates a set of disjoint complete graphs.
  Parameters:
    orders ({[],()}) : the orders (number of nodes of the graphs)
  Returns:
    g (nx.Graph) : a disjoint graph
  """

  if not all([isinstance(x, int) for x in orders]):
    raise TypeError("Orders must be integers")

  if not all(orders):
    raise ValueError("Number of nodes must be greater than zero. Got {0}".format(orders))

  g = nx.disjoint_union_all([nx.complete_graph(n_nodes) for n_nodes in orders])

  return g

#------------------------------------------
def LinkedCliques(orders, link_density):
  """
  Creates a set of cliques that are linked to each other.
  Parameters:
    orders ([int]) : orders of the initially disjoint cliques
    link density ({float,[float]}) : the ratio of edges with respect to the average number of nodes in the cliques.
    If float, the link density between nodes are uniform. 
    If list of floats, the list should be orders*(order+1) length.
  """

  if not isinstance(orders, float):
    pass