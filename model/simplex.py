import torch
import itertools
from .g_sub_to_g_ind import g_sub_to_g_ind
from .ind2subv import ind2subv

import torch

def belief_simplex(G, k):
    """
    Regular grid to belief simplex construction in PyTorch.

    Args:
        G: points on regular simplex returned by regular_simplex (tensor of size [dim, num_points])
        k: grid size

    Returns:
        B: points in belief simplex (tensor of size [dim, num_points])
        B_to_G: transformation matrix from belief to grid
        G_to_B: transformation matrix from grid to belief
    """
    
    dim = G.shape[0]

    # Construct the upper triangular matrix B_to_G
    B_to_G = torch.triu(torch.full((dim, dim), k, dtype=torch.float64))

    # Invert the B_to_G matrix to get G_to_B
    G_to_B = torch.inverse(B_to_G)

    # Multiply G_to_B with G to get points in the belief simplex
    B = torch.matmul(G_to_B, G.to(torch.float64))

    return B, B_to_G, G_to_B


def regular_simplex(dim, k):
    """
    Constructs a regular simplex grid.

    Parameters:
        dim (int): Number of dimensions.
        k (int): Grid size -- each edge of the simplex has k+1 points.

    Returns:
        G_sub (torch.Tensor): Subscripts of the grid points in the simplex of shape [dim, N].
        G_ind (torch.Tensor): Linear indices of the grid points in the simplex of shape [N].
    """
    # Generate all points on the hypercube
    N = (k + 1) ** (dim - 1)
    indices = torch.arange(1, N + 1, dtype=torch.long)  # 1-based indexing

    sizes = [k + 1] * (dim - 1)
    
    G = ind2subv(sizes, indices).T

    
    # Construct G_sub
    G_sub = torch.cat([torch.full((1, G.size(1)), k), G - 1], dim=0)
    
    # Apply lower triangular constraint
    diff_G_sub = G_sub[1:, :] - G_sub[:-1, :]
    valid_mask = torch.all(diff_G_sub <= 0, dim=0)
    G_ind_valid = torch.nonzero(valid_mask).squeeze()

    
    # Filter valid subscripts
    G_sub = G_sub[:, G_ind_valid]

    # Compute linear indices

    G_ind = g_sub_to_g_ind(G_sub, dim, k)
    print(G_sub)
    return G_sub, G_ind
