"""
Base class of Markov cluster classifier.
"""

#-----------------------
class BaseMC(object):
  """
  Base class of the Markov cluster classifier.
  Attributes:
    diag_scale (float>0) : factor for scaling the diagonal of the connectivity matrix

    expand_power (int) : number of Markov process steps

    inflate_power {int, float} : the normalising parameter for the transition probabilities. Default 2.

    max_iter (int) : maximum number of iterations. Default 10. max_iter * expand_power is is the total number of Markov steps.

    threshold (float) : elements below this value will be set to zero. Default 10^{-5}.

    tol (float) : Terminating criterion. If the L1 variance of the nonzero elements fall below tol,
        the Markov process is terminated. Default 10^{-3}

  Methods:
    get_params() : returns the parameters of the estimator
    set_params() : set the parameters of the estimator
  """

#-----------
  def __init__(self, expand_power = 2, inflate_power = 2, max_iter = 10, threshold = 0.00001, tol = 0.001):
    """
    Initialises an instance of the Markoc cluster algorithm
    Parameters:
      expand_power ({int, float}) : the number of Markoc process steps. Default 2.
      inflate_power {int, float} : the normalising parameter for the transition probabilities. Default 2.
      max_iter (int) : maximum number of iterations. Default 10. max_iter * expand_power is is the total number of Markov steps.
      threshold (float) : elements below this value will be set to zero. Default 10^{-5}.
      tol (float) : Terminating criterion. If the L1 variance of the nonzero elements fall below tol, the Markov process is terminated. Default 10^{-3}
      """

    if diag_scale <= 0.0:
      raise ValueError("diag_scale must be positive")
    self._diag_scale = diag_scale

    if not isinstance(expand_power, int):
      raise TypeError("expand_power must be of type int. Got: {0}".format(type(expand_power)))

    if expand_power <= 0:
      raise ValueError
    self._expand_power = expand_power

    if inflate_power <= 0:
      raise ValueError("inflate_power must be positive")
    self._inflate_power = inflate_power

    if max_iter < 1:
      raise ValueError("max_iter must be positive")
    self._max_iter = max_iter

    if tol < 0.0:
      raise ValueError("tol must be positive")
    self._tol = tol

    self._params = {'diag_scale' : self.diag_scale, 
                    'expand_power' : self.expand_power,
                    'inflate_power' : self.inflate_power,
                    'max_iter' : self.max_iter,
                    'threshold' : self.threshold,
                    'tol' : self.tol}

  @property
  def diag_scale(self):
    return self._diag_scale

  @property
  def expand_power(self):
    return self._expand_power

  @property
  def inflate_power(self):
    return self._inflate_power
    
  @property
  def max_iter(self):
    return self._max_iter

  @property
  def threshold(self):
    return self._threshold

  @property
  def tol(self):
    return self._tol

#-----------
  def get_params(self):
    """
    Returns a dictionary of parameters. 
    Parameters:
      self
    Returns:
      param_dict ({}) : dictionary of parameters
    """

# reconstruct dict in order to protect parameters
    param_dict = dict(self._params.items())

    return param_dict

#-----------
  def set_params(self, param_dict):
    """
    Sets the parameters of the estimator.
    Parameters:
      param_dict : a dictionary containing the key value pairs of the parameters.
    """
    param_names = self.get_params().keys()

    for _pname, _pval in param_dict.items():
      if _pname not in param_names:
        raise KeyError("Invalid keyword {0}".format(_pname))

      self._params[_pname] = _pval

#----------
  def fit(self, X):
    """
    Fits clusters to the graph using Markov cluster algorithm.
    Parameters:
      X (np.ndarray[n_nodes, n_nodes]) : 2D connectivity matrix
    """

    raise NotImplementedError("Fit has not been implemented in base class")
















