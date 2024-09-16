import torch
import numpy as np
from model import ind2subv

def create_goal_reward(worlds, n_reward_grid):
    """
    Create goal reward matrix.
    
    Parameters:
    - worlds: A list of world objects, where each world contains a 'goal_pose' attribute.
    - n_reward_grid: The number of reward grids.
    
    Returns:
    - goal_reward: A tensor containing the goal rewards.
    """
    # Get the number of goal objects (assumed to be the same for all worlds)

    n_goal_obj = len(worlds[0]['goal_pose'])


    # Generate indices for the goal rewards
    indices = torch.arange(1, (n_reward_grid ** n_goal_obj) + 1)
    G = ind2subv([n_reward_grid] * n_goal_obj, indices).float() - 1  # Convert linear indices to subscripts and subtract 1

    # Calculate goal rewards
    goal_reward = -20 + G * 20
    
    return goal_reward



