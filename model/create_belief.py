import torch
from .simplex import regular_simplex
from .simplex import belief_simplex, belief_simplex

def create_belief_space(n_world, n_grid):
    """
    Create belief space in PyTorch.

    Args:
        n_world: number of possible worlds (vertices on simplex)
        n_grid: each edge of the belief simplex has n_grid+1 points

    Returns:
        b_sub: belief sub grid
        b_to_g: mapping from belief to grid
        g_to_b: mapping from grid to belief
    """
    
    if n_grid > 0:
        g_sub, g_ind = regular_simplex(n_world, n_grid)
        g_to_b = torch.zeros((1, (n_grid+1) ** n_world), dtype=torch.long)
        
        g_to_b[g_ind] = torch.arange(1, len(g_ind) + 1)


        b_sub, b_to_g = belief_simplex(g_sub, n_grid)

    else:
        b_sub = torch.tensor([[1.0], [1.0], [1.0]]) / 3
        g_to_b = torch.ones((n_world, n_world)) / 9
        b_to_g = torch.ones((n_world, n_world))

    return b_sub, b_to_g, g_to_b

def create_belief_state(is_c_ind_valid, n_grid):
    """
    Create belief state in PyTorch

    Args:
        is_c_ind_valid: tensor of size (n_world, n_c_ind)
        n_grid: each edge of the belief simplex has n_grid+1 points

    Returns:
        s_sub (state_sub): [b_ind, c_ind]
        s_dim (state_dim): [n_b_sub, n_c_ind]
        b_sub: tensor of size (n_world, n_b_sub)
        b_to_g: tensor of size (n_world, n_world)
        g_to_b: tensor of size (n_world, n_world)
    """
    
    n_world, n_c_ind = is_c_ind_valid.shape

    print(n_world, n_c_ind)

    # Create belief space
    b_sub, b_to_g, g_to_b = create_belief_space(n_world, n_grid)
    n_b_sub = b_sub.shape[1]

    # Create belief state
    s_dim = torch.tensor([n_b_sub, n_c_ind])
    n_s_ind = torch.prod(s_dim).item()
    s_ind = torch.arange(1, n_s_ind + 1)

    # Equivalent of ind2subv in MATLAB
    s_sub = torch.stack(torch.meshgrid([torch.arange(d) for d in s_dim], indexing='ij'), dim=-1).reshape(-1, len(s_dim))

    return s_sub, s_dim, b_sub, b_to_g, g_to_b