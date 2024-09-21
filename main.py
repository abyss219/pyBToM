from model import create_belief_state_sptrans
import torch
import numpy as np
from utils import equals

# Test cases for Python code
s_dim = [2, 2]
s_sub = np.array([[1, 2], [1, 2]])
w_trans = np.array([[0.7, 0.3], [0.4, 0.6]])
c_trans = np.random.rand(2, 2, 2, 2)
obs_dist = np.random.rand(2, 2)
b_sub = np.random.rand(2, 2)
b_to_g = np.random.rand(2, 2)
g_ind_to_b_ind = np.array([1, 2])

s_trans_python, s_trans_ind_python = create_belief_state_sptrans(s_dim, s_sub, w_trans, c_trans, obs_dist, b_sub, b_to_g, g_ind_to_b_ind)