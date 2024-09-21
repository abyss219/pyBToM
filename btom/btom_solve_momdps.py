import torch
import numpy as np
import scipy.io
from .create_goal_reward import create_goal_reward
from .config import *
from utils import convert_mat_data, equals, length
from model import create_coord_trans, create_belief_state, create_belief_state_sptrans

def btom_solve_momdps(beta_score):
    # Load stimuli (worlds) and model parameters
    data = scipy.io.loadmat('data/stimuli.mat', simplify_cells=True)
    worlds = data['worlds']
    worlds = convert_mat_data(worlds)
    
    if visilibity:
        obs_dist = [None] * n_worlds
    else:
        obs_dist = scipy.io.loadmat('data/visilibity/observation_distribution.mat', simplify_cells=True)['obs_dist']
        obs_dist = convert_mat_data(obs_dist)
        obs_dist = np.array(obs_dist)


    

    n_worlds = len(worlds)
    n_worlds = 1
    for nw in range(n_worlds):
        print(f"Processing world {nw+1}/{n_worlds}")
        goal_reward = create_goal_reward(worlds[nw], n_reward_grid)

        n_goal_reward = goal_reward.shape[0]
        c_trans, c_sub, is_c_ind_valid = create_coord_trans(worlds[nw], action, move_noise)

        
        n_world = length(np.array(worlds[nw]))
        w_trans = np.eye(n_world)
        s_sub, s_dim, b_sub, b_sub_to_g_sub, g_ind_to_b_ind = create_belief_state(is_c_ind_valid, n_belief_grid)
        
        s_trans,s_trans_ind = create_belief_state_sptrans(s_dim,s_sub,w_trans,c_trans,obs_dist[nw],b_sub,b_sub_to_g_sub,g_ind_to_b_ind)




