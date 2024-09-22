import numpy as np
from utils import length, find

def create_state_reward(world, c_sub, is_c_ind_valid, goal_reward, cost):
    """
    Create goal-based reward function for world state with goal-achieved ("finished") bit.
    Set up "finished" transition function for each world.

    Input:
      - world: list of dictionaries
      - c_sub: numpy array of coordinates (dimensions x n_c_sub)
      - is_c_ind_valid: numpy array (n_world x n_c_ind)
      - goal_reward: numpy array (n_goal_ind x n_goal_obj)
      - cost: list or numpy array of action costs

    Output:
      - reward: numpy array (n_s_ind x n_action)
    """
    n_world, n_c_ind = is_c_ind_valid.shape
    n_c_sub = c_sub.shape[1]
    n_action = length(cost)
    n_goal_ind, n_goal_obj = goal_reward.shape

    s_dim = [n_world, n_c_ind]
    n_s_ind = np.prod(s_dim)
    s_ind = np.arange(n_s_ind)
    # s_sub = ind2subv(s_dim, s_ind)  # Not used in the function

    wcg_reward = np.zeros((n_world, n_c_ind, n_goal_ind))

    for wi in range(n_world):
        for go in range(n_goal_obj):
            if len(world[wi]['goal_pose'][go]) > 0:

                goal_pose = np.array(world[wi]['goal_pose'][go])
                
                goal_pose = np.tile(goal_pose[:, np.newaxis], [1, n_c_sub])
                

                comparison = (goal_pose == c_sub)
                match_ = np.all(comparison, axis=0)
                valid = is_c_ind_valid[wi, :n_c_sub]
                overall_condition = match_ & valid
                goal_obj_c_ind = find(overall_condition)

                if goal_obj_c_ind.size > 0:
                    for gi in range(n_goal_ind):
                        wcg_reward[wi, goal_obj_c_ind, gi] += goal_reward[gi, go]

    cost = cost.astype(np.float64)
    reward = np.tile(cost, (n_s_ind, 1))
    wcg_reward_sum = wcg_reward.sum(axis=2)
    
    reward[:, -1] += wcg_reward_sum.flatten(order='F')

    return reward