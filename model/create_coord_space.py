import torch
from .ind2subv import ind2subv
from .sub2indv import sub2indv
from utils import equals

def create_coord_space(world):
    """
    Create deterministic coordinate transition matrix using PyTorch.
    
    Input:
        - world: list of world objects
    
    Output:
        - c_sub: matrix of coordinate subscripts
        - is_c_ind_valid: boolean matrix indicating valid coordinate indices
    """
    n_world = len(world)

    # Compute c_sub and is_c_ind_valid
    # Assuming all worlds have the same graph_sz as world[0]
    graph_sz = world[0]['graph_sz']
    n_c_sub = torch.prod(torch.tensor(graph_sz))

    # Generate c_sub: subscripts for all coordinates
    c_sub = ind2subv(graph_sz, torch.arange(1, n_c_sub + 1)).T

    # Initialize is_c_ind_valid: True means valid, False means invalid (obstacle)
    n_c_ind = n_c_sub + 1
    is_c_ind_valid = torch.ones((n_world, n_c_ind), dtype=torch.bool)

    # Check for obstacles and mark invalid indices
    for nw in range(n_world):
        obst_pose = world[nw]['obst_pose']
        obst_sz = world[nw]['obst_sz']
        obst_sub = get_obst_sub(obst_pose, obst_sz)
        obst_ind = sub2indv(graph_sz, obst_sub) - 1
        is_c_ind_valid[nw, obst_ind] = False

    return c_sub, is_c_ind_valid


def get_obst_sub(obst_pose, obst_sz):
    """
    Get obstacle subscript positions based on obstacle size and position using PyTorch.
    
    Input:
        - obst_pose: Tensor of shape (2, n_obst) or (2,) if n_obst == 1
        - obst_sz: Tensor of shape (2, n_obst) or (2,) if n_obst == 1
    
    Output:
        - obst_sub: Tensor of shape (2, total_obstacle_indices)
    """
    # Convert inputs to tensors if they aren't already
    obst_pose = torch.tensor(obst_pose, dtype=torch.int32)
    obst_sz = torch.tensor(obst_sz, dtype=torch.int32)

    # Handle empty obst_pose
    if obst_pose.numel() == 0:
        obst_sub = torch.zeros((2, 0), dtype=torch.int32)
        return obst_sub

    # Ensure obst_pose and obst_sz are 2D tensors
    if obst_pose.ndim == 1:
        obst_pose = obst_pose.unsqueeze(1)
    if obst_sz.ndim == 1:
        obst_sz = obst_sz.unsqueeze(1)

    n_obst = obst_pose.shape[1]
    if n_obst > 0:
        n_ind = torch.prod(obst_sz, dim=0)
        total_ind = n_ind.sum().item()
        obst_sub = torch.zeros((2, total_ind), dtype=torch.int32)
        ind = 0
        for oi in range(n_obst):
            sz_x = obst_sz[0, oi].item()
            sz_y = obst_sz[1, oi].item()
            for xi in range(sz_x):
                for yi in range(sz_y):
                    obst_sub[:, ind] = obst_pose[:, oi] + torch.tensor([xi, yi], dtype=torch.int32)
                    ind += 1
    else:
        obst_sub = torch.zeros((2, 0), dtype=torch.int32)
    
    return obst_sub