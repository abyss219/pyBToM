import torch
import scipy.io
from .create_goal_reward import create_goal_reward
from .config import *
from utils import convert_mat_data
from model import create_coord_trans, create_belief_state

def btom_solve_momdps(beta_score):
    # Load stimuli (worlds) and model parameters
    data = scipy.io.loadmat('data/stimuli.mat', simplify_cells=True)
    worlds = data['worlds']
    worlds = convert_mat_data(worlds)

    

    n_worlds = len(worlds)

    for nw in range(n_worlds):
        print(f"Processing world {nw+1}/{n_worlds}")
        goal_reward = create_goal_reward(worlds[nw], n_reward_grid)
        n_goal_reward = goal_reward.shape[0]
        c_trans, c_sub, is_c_ind_valid = create_coord_trans(worlds[nw], action, move_noise)
        n_world = len(worlds[nw])
        w_trans = torch.eye(n_world)
        s_sub, s_dim, b_sub, b_sub_to_g_sub, g_ind_to_b_ind = create_belief_state(is_c_ind_valid, n_belief_grid)




