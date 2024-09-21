import torch
import itertools
from .g_sub_to_g_ind import g_sub_to_g_ind
from .ind2subv import ind2subv
from utils import find

import torch

import numpy as np

def belief_simplex(G, k):
    """
    lovejoy regular grid -> belief simplex construction

    Input:
        G: points on regular simplex returned by regular_simplex
        k: grid size

    Output:
        B: Transformed points on the belief simplex
        B_to_G: Upper triangular matrix used for transformation from B to G
        G_to_B: Inverse of B_to_G, used for transformation from G to B
    """
    dim = G.shape[0]
    B_to_G = np.triu(np.full((dim, dim), k))
    G_to_B = np.linalg.inv(B_to_G)
    B = G_to_B @ G.astype(float)

    return B, B_to_G, G_to_B


def regular_simplex(dim, k):
    # Generate all points on the hypercube
    G = ind2subv(np.tile(k+1, (dim-1,)), np.arange(1, (k+1)**(dim-1) + 1)).T

    # Create G_sub by adding the k row and adjusting G
    G_sub = np.vstack([np.tile(k, (1, G.shape[1])), G - 1])

    # Apply the lower triangular constraint (all(diff(G_sub) <= 0, axis=1) in MATLAB)
    diffs = np.diff(G_sub, axis=0) <= 0
    G_ind_valid = find(np.all(diffs, axis=0))

    # Subset G_sub with valid indices
    G_sub = G_sub[:, G_ind_valid]

    # Convert G_sub to G_ind using the g_sub_to_g_ind function
    G_ind = g_sub_to_g_ind(G_sub, dim, k)
    return G_sub, G_ind