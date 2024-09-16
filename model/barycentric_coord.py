import torch
from .g_sub_to_g_ind import g_sub_to_g_ind

def barycentric_coord(b_sub, b_to_g):
    # Unique rows in b_sub (transpose for similar behavior as MATLAB)
    b_sub_t = b_sub.T  # No need to convert to tensor, assuming input is already a tensor
    b_sub_u, b_ind = torch.unique(b_sub_t, dim=0, return_inverse=True)
    
    # Size of unique subscript set
    n_b_sub, n_dim = b_sub_u.size()

    # Compute g_sub
    g_sub = torch.matmul(b_to_g, b_sub_u.T)

    # Fix base and compute d
    base = torch.floor(g_sub)  # Using floor to mimic fix() in MATLAB
    d = g_sub - base

    # Initialize variables
    v = torch.zeros(n_dim, n_dim, n_b_sub, dtype=torch.float32)
    Id = torch.eye(n_dim, dtype=torch.float32)
    b_coord = torch.zeros(n_dim, n_b_sub, dtype=torch.float32)

    neighbor_g_ind = torch.zeros(n_dim, n_b_sub, dtype=torch.int32)
    k = int(b_to_g[0, 0])
    
    for ni in range(n_b_sub):
        # Sort d in descending order
        d_sort, d_order = torch.sort(d[:, ni], descending=True)
        
        # Compute neighboring vertices
        v[:, 0, ni] = base[:, ni]
        for nd in range(n_dim - 1):
            v[:, nd + 1, ni] = v[:, nd, ni] + Id[:, d_order[nd]]
        
        # neighbor_g_ind is a sub-function, assumed to be defined elsewhere
        neighbor_g_ind[:, ni] = g_sub_to_g_ind(v[:, :, ni], n_dim, k)

        # Compute barycentric coordinates
        b_coord[n_dim - 1, ni] = d_sort[n_dim - 2]
        for nd in range(n_dim - 2, 0, -1):
            b_coord[nd, ni] = d_sort[nd - 1] - d_sort[nd]

    # Final coordinate adjustment
    b_coord[0, :] = 1 - torch.sum(b_coord[1:, :], dim=0)
    b_coord = torch.maximum(b_coord, torch.tensor(0.0))  # Protect against numerical error

    # Reindex to original ordering
    b_coord = b_coord[:, b_ind]
    neighbor_g_ind = neighbor_g_ind[:, b_ind]

    return neighbor_g_ind, b_coord
