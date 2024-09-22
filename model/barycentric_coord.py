import numpy as np
from .g_sub_to_g_ind import g_sub_to_g_ind
from utils import equals

def barycentric_coord(b_sub, b_to_g):
    """
    Input:
      - b_sub: Input subscript array.
      - b_to_g: Transformation matrix.
    
    Output:
      - neighbor_g_ind: Neighbor global indices.
      - b_coord: Barycentric coordinates.
    """
    # Transpose b_sub and find unique rows
    b_sub_t = b_sub.T
    b_sub_u, u_ind = np.unique(b_sub_t, axis=0, return_inverse=True)
    n_b_sub, n_dim = b_sub_u.shape


    # Compute g_sub
    g_sub = np.dot(b_to_g, b_sub_u.T)

    # Compute base and difference
    base = np.fix(g_sub.astype(np.float32))
    d = g_sub - base

    # Initialize variables
    v = np.zeros((n_dim, n_dim, n_b_sub))
    Id = np.eye(n_dim)
    b_coord = np.zeros((n_dim, n_b_sub))
    neighbor_g_ind = np.zeros((n_dim, n_b_sub))
    k = b_to_g.flatten(order='F')[0]

    for ni in range(n_b_sub):
        d_ni = d[:, ni]
        d_order = np.argsort(-d_ni)  # Descending sort
        d_sort = d_ni[d_order]

        # Compute neighboring vertices
        v[:, 0, ni] = base[:, ni]
        
        for nd in range(n_dim - 1):
            v[:, nd + 1, ni] = v[:, nd, ni] + Id[:, d_order[nd]]
        neighbor_g_ind[:, ni] = g_sub_to_g_ind(v[:, :, ni], n_dim, k)
        

        # Compute barycentric coordinates
        b_coord[n_dim - 1, ni] = d_sort[n_dim - 2]
        for nd in range(n_dim - 2, 0, -1):
            b_coord[nd, ni] = d_sort[nd - 1] - d_sort[nd]
    b_coord[0, :] = 1 - np.sum(b_coord[1:, :], axis=0)
    b_coord = np.maximum(b_coord, 0)  # Protect against numerical error

    # Reorder according to u_ind
    b_coord = b_coord[:, u_ind]
    neighbor_g_ind = neighbor_g_ind[:, u_ind]

    return neighbor_g_ind, b_coord

