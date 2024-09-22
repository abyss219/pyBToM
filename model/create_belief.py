from .simplex import regular_simplex, belief_simplex
from .ind2subv import ind2subv
import numpy as np

def create_belief_space(n_world, n_grid):
    """
    Input:
        n_world: number of possible worlds (vertices on simplex)
        n_grid: each edge of the belief simplex has n_grid+1 points

    Output:
        b_sub
        b_to_g
        g_to_b
    """
    if n_grid > 0:
        g_sub, g_ind = regular_simplex(n_world, n_grid)
        g_to_b = np.zeros(((n_grid + 1) ** n_world,), dtype=np.uint32)
        g_to_b[g_ind - 1] = np.arange(1, len(g_ind) + 1)
        b_sub, b_to_g, _ = belief_simplex(g_sub, n_grid)
    else:
        b_sub = np.array([[1], [1], [1]]) / 3.0
        g_to_b = np.ones((n_world, n_world)) / 9.0
        b_to_g = np.ones((n_world, n_world))
    return b_sub, b_to_g, g_to_b


def create_belief_state(is_c_ind_valid, n_grid):
    """
    Input:
      is_c_ind_valid: shape (n_world, n_c_ind)
      n_grid: each edge of the belief simplex has n_grid+1 points

    Output:
      - s_sub (state_sub): [b_ind, c_ind]
      - s_dim (state_dim): [n_b_sub, n_c_ind]
      - b_sub
      - b_to_g
      - g_to_b
    """

    n_world, n_c_ind = is_c_ind_valid.shape

    # Create belief space
    b_sub, b_to_g, g_to_b = create_belief_space(n_world, n_grid)

    n_b_sub = b_sub.shape[1]

    # Create belief state
    s_dim = np.array([n_b_sub, n_c_ind])
    
    n_s_ind = np.prod(s_dim)
    s_ind = np.arange(n_s_ind) + 1  # Zero-based indexing
    s_sub = ind2subv(s_dim, s_ind).T

    return s_sub, s_dim, b_sub, b_to_g, g_to_b