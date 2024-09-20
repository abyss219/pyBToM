import numpy as np
from .ind2subv import ind2subv
from .sub2indv import sub2indv
from utils import equals

def create_coord_space(world):
    """
    Create deterministic coordinate transition matrix.

    Input:
        world: List of dictionaries, each representing a world with 'graph_sz',
               'obst_pose', and 'obst_sz' keys.

    Output:
        c_sub: Coordinate subscripts.
        is_c_ind_valid: Boolean array indicating valid indices.
    """
    n_world = len(world)
    graph_sz = np.array(world[0]['graph_sz'])
    n_c_sub = np.prod(graph_sz)
    c_sub = ind2subv(graph_sz, np.arange(1, n_c_sub + 1)).T.astype(float)


    n_c_ind = n_c_sub + 1
    is_c_ind_valid = np.ones((n_world, n_c_ind), dtype=bool)
    for nw in range(n_world):
        obst_pose = np.array(world[nw]['obst_pose'])
        obst_pose = obst_pose[:, np.newaxis]
        obst_sz = np.array(world[nw]['obst_sz'])
        obst_sz = obst_sz[:, np.newaxis]
        obst_sub = get_obst_sub(obst_pose, obst_sz)
        obst_ind = sub2indv(graph_sz, obst_sub)
        is_c_ind_valid[nw, obst_ind - 1] = False
    return c_sub, is_c_ind_valid

def get_obst_sub(obst_pose, obst_sz):
    """
    Get obstacle subscripts.

    Input:
        obst_pose: 2D NumPy array of obstacle positions.
        obst_sz: 2D NumPy array of obstacle sizes.

    Output:
        obst_sub: 2D NumPy array of obstacle subscripts.
    """
    n_obst = obst_pose.shape[1]
    if n_obst > 0:
        n_ind = np.sum(obst_sz[0, :] * obst_sz[1, :])
        obst_sub = np.zeros((2, n_ind), dtype=int)
        ind = 0
        for oi in range(n_obst):
            for xi in range(obst_sz[0, oi]):
                for yi in range(obst_sz[1, oi]):
                    obst_sub[:, ind] = obst_pose[:, oi] + np.array([xi, yi])
                    ind += 1
    else:
        obst_sub = np.zeros((2, 0), dtype=int)
    return obst_sub