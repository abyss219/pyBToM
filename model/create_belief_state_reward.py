import numpy as np
from .create_state_reward import create_state_reward
from utils import length

def create_belief_state_reward(world, s_sub, s_dim, b_sub, c_sub, is_c_ind_valid, goal_reward, cost):
    # [reward] = create_belief_state_reward(world,s_sub,s_dim,b_sub,c_sub,is_c_ind_valid,goal_reward,cost)
    #
    # Input:
    #   - world
    #   - s_sub
    #   - s_dim
    #   - b_sub
    #   - c_sub
    #   - is_c_ind_valid
    #   - goal_reward
    #   - cost
    #
    # Output:
    #   - reward
    #   - s_sub (state_sub): [belief_ind; coord_ind]
    #   - s_dim (state_dim): [n_b_sub; n_c_sub]
    #
    #
    # n_world = length(world);
    n_world = length(world)

    # n_s_ind = prod(s_dim);
    n_s_ind = np.prod(s_dim)

    # [n_b_sub,n_c_sub] = deal(s_dim(1),s_dim(2));
    n_b_sub, n_c_sub = s_dim[0], s_dim[1]

    # n_action = size(cost,2);
    n_action = cost.shape[1]

    # get fully observable reward function
    # ws_reward = create_state_reward(world,c_sub,is_c_ind_valid,goal_reward,cost);
    ws_reward = create_state_reward(world, c_sub, is_c_ind_valid, goal_reward, cost)

    # create belief state reward
    # tmp1 = reshape(repmat(reshape(ws_reward,[n_world 1 n_c_sub,n_action]),[1 n_b_sub 1 1]),[n_world,n_s_ind,n_action]);
    tmp1 = np.reshape(ws_reward, (n_world, 1, n_c_sub, n_action), order='F')
    tmp1 = np.tile(tmp1, (1, n_b_sub, 1, 1))
    tmp1 = np.reshape(tmp1, (n_world, n_s_ind, n_action), order='F')

    # tmp2 = repmat(b_sub(:,s_sub(1,:)),[1 1 n_action]);
    # Adjust for MATLAB 1-based indexing
    indices = s_sub[0, :] - 1  # Convert to zero-based indexing
    tmp2 = b_sub[:, indices]  # Shape: (n_world, n_s_ind)
    tmp2 = np.tile(tmp2[:, :, np.newaxis], (1, 1, n_action))  # Shape: (n_world, n_s_ind, n_action)

    # reward = squeeze(sum(tmp1.*tmp2,1));
    reward = np.sum(tmp1 * tmp2, axis=0)  # Sum over n_world dimension
    return reward
