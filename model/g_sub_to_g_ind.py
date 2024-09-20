import numpy as np
from .sub2indv import sub2indv

def g_sub_to_g_ind(g_sub, dim, k):
    """
    [g_ind] = g_sub_to_g_ind(g_sub, dim, k)
    
    Converts subscripts to linear indices, ensuring that subscripts are within the valid range.
    
    Parameters:
    g_sub (ndarray): Subscripts array of shape (dim, N)
    dim (int): Number of dimensions
    k (int): Maximum allowed subscript value in each dimension
    
    Returns:
    g_ind (ndarray): Linear indices corresponding to the valid subscripts
    """
    # Initialize g_ind with zeros
    g_ind = np.zeros(g_sub.shape[1], dtype=int)
    
    # Determine which columns are within the valid range [0, k]
    g_ind_in_range = np.all((g_sub <= k) & (g_sub >= 0), axis=0)
    
    # Prepare dimensions array
    dims = np.full(dim, k + 1)

    
    # Get subscripts for columns that are in range
    subs = g_sub[:, g_ind_in_range] + 1
    
    # Compute linear indices for valid subscripts
    g_ind[g_ind_in_range] = sub2indv(dims, subs)
    
    return g_ind


