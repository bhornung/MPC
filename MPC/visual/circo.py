import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

colours = {0 : np.array((1, 0, 0)),
           1 : np.array((0, 1, 0)),
           2 : np.array((0, 0, 1)),
           3 : np.array((1, 0, 1))}

def _get_colour_from_circle(x):
    """
    Returns an RGB colour from the rim of the colour circle at x * 2 * \pi.
    """
    base_cix, mixing = np.divmod(x * 3, 1)
    base_cix = int(base_cix)
    colour = colours[base_cix] * (1.0 - mixing) + colours[base_cix + 1] * mixing
    return tuple(colour)

def _bezier(pos_start, post_end, pos_control, n_points):
    """
    Calculates the quadratic Bezier curve given the points.
    """
    t = np.linspace(0, 1, num = n_points)[:,None]
    s = (1.0 - t)  
    pos = s * (s * pos_start + t * pos_control) + t * (s * pos_control + t * post_end)
    pos = pos.T

    return pos

def _set_circle_positions(n_points):
    """
    Calculates the coordinates of a given number of points on the unit circle
    Parameters:
        n_points (int) : number of points
    Returns
        pos (np.ndarray (2, npoints), np.float) : positions of the points.
    """

    angles = np.linspace(0, np.pi * 2, num = n_points, endpoint = False)
    pos = np.vstack([np.cos(angles), np.sin(angles)]).T

    return pos

def _set_control_pos(pos_start, pos_end, pos_centre = np.zeros(2, dtype = np.float), scale = 1.0):
    """
    Sets the control point of a quadratic Bezier curve.
    Parameters:
        pos_start (np.ndarray[]) : starting point of the Bezier curve
        pos_end (np.ndarray[]) : end point of the Bezier curve
        pos_centre (np.ndarray[]) : reference point
        scale (float) : sets radius of the arc. 0.0: straight line, R, 1.0: R_{circ}. Default 1.0.
    Returns:
        pos_control (np.ndarray[]) : control point of the Bezier curve
    """
    _mid = (pos_start + pos_end) * 0.50
    pos_control = scale * pos_centre + (1.0 - scale) * _mid 

    return pos_control

def plot_circo(mat, ax = None, cutoff = 0.01, n_arc_points = 10, radius = 1.0, color = 'red'):
    """

    """
# --- preps
    mat_ = sps.coo_matrix(mat)
    mask = mat_.data > cutoff
    mat_.data = mat_.data[mask]
    mat_.row = mat_.row[mask]
    mat_.col = mat_.col[mask]

    n_nodes = mat_.shape[1]

# --- plot circle
    circle = _set_circle_positions(n_nodes) * radius

    if ax is None:
        plt.scatter(circle[:,0], circle[:,1], marker = 'o', color = 'black', facecolors = 'none')
    else:
        ax.scatter(circle[:,0], circle[:,1], marker = 'o', color = 'black', facecolors = 'none')

# --- create Bezier curves for all edges
    for i_source, i_target, weight in zip(mat_.row, mat_.col, mat_.data):
# get defining points of the arc
        pos_start, pos_end = circle[i_source], circle[i_target]
        pos_control = _set_control_pos(pos_start, pos_end, scale = 0.5)
        arc = _bezier(pos_start, pos_end, pos_control, n_arc_points)

        linewidth = np.power(weight, 0.8)

# set colour
        colour_circ_pos = i_target * 1.0 / n_nodes
        #colour_circ_pos -= 0.17
        if colour_circ_pos < 0.0:
            colour_circ_pos = 1.0 + colour_circ_pos

        color = _get_colour_from_circle(colour_circ_pos)
# plot arc
        if ax is None:
            plt.plot(arc[0], arc[1], color = color, linewidth = linewidth)
        else:
            ax.plot(arc[0], arc[1], color = color, linewidth = linewidth)


#import networkx as nx
#mat = nx.adjacency_matrix(nx.random_partition_graph([3,4,5,6], 1.0, 0.1, seed = 42)).tocsr()
#plot_circo(mat)
#plt.show()

