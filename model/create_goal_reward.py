import numpy as np
from .ind2subv import ind2subv
from utils import length, pad_sublists

def create_goal_reward(worlds, n_reward_grid):
    """
    [goal_reward] = create_goal_reward(worlds, n_reward_grid)
    
    Generates the goal reward matrix based on the number of reward grids.
    
    Parameters:
    - worlds: list of world objects or dictionaries with a 'goal_pose' attribute
    - n_reward_grid: integer, number of reward grids
    
    Returns:
    - goal_reward: numpy array of goal rewards
    """
    # Ensure n_reward_grid is a positive integer
    if not isinstance(n_reward_grid, int) or n_reward_grid <= 0:
        raise ValueError("n_reward_grid must be a positive integer.")

    p  = pad_sublists(worlds[0]['goal_pose'])
    p = np.array(p)
    n_goal_obj = length(p)

    # Create size array: [n_reward_grid, n_reward_grid, ..., n_goal_obj times]
    siz = [n_reward_grid] * n_goal_obj

    # Total number of indices
    total_indices = n_reward_grid ** n_goal_obj

    # Create index array: from 1 to total_indices inclusive (1-based indexing)
    index = np.arange(1, total_indices + 1, dtype=np.uint32)

    # Compute G: subscript vectors corresponding to linear indices in an array of size siz
    G = ind2subv(siz, index) - 1  # Subtract 1 to adjust to zero-based indexing
    G = G.astype(np.float64)      # Convert to double precision float

    # Compute goal_reward
    goal_reward = -20 + G * 20    # Element-wise multiplication and addition

    return goal_reward
