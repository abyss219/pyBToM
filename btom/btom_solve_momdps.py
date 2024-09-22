import os
import numpy as np
import scipy.io
from .config import *
from model import *
from utils import convert_mat_data, length


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
    for nw in range(n_worlds):
        print(f"Processing world {nw+1}/{n_worlds}")
        goal_reward = create_goal_reward(worlds[nw], n_reward_grid)

        n_goal_reward = goal_reward.shape[0]
        c_trans, c_sub, is_c_ind_valid = create_coord_trans(worlds[nw], action, move_noise)

        
        n_world = length(np.array(worlds[nw]))
        w_trans = np.eye(n_world)
        s_sub, s_dim, b_sub, b_sub_to_g_sub, g_ind_to_b_ind = create_belief_state(is_c_ind_valid, n_belief_grid)
        
        s_trans,s_trans_ind = create_belief_state_sptrans(s_dim,s_sub,w_trans,c_trans,obs_dist[nw],b_sub,b_sub_to_g_sub,g_ind_to_b_ind)
        # equals(s_trans_ind, True)

        mdp_options['trans_ind'] = s_trans_ind

        savedir = 'output'
        os.makedirs(savedir, exist_ok=True)

        for ng in range(n_goal_reward):
            s_reward = create_belief_state_reward(worlds[nw],s_sub,s_dim,b_sub,c_sub,is_c_ind_valid,goal_reward[ng,:][np.newaxis, :],np.array(action_cost)[np.newaxis, :])
            Q,V,n_iter,err = mdp_Q_VI([],s_trans,s_reward,mdp_options)
            filename = f"{savedir}/value{ng:03d}.npz"
            np.savez(filename, Q=Q, V=V)
