from model import create_state_reward
import torch
import numpy as np
from utils import equals

# Test Case 1: Single World, Single Coordinate, Single Goal
world = [{'goal_pose': [[np.array([1, 1])]]}]
c_sub = np.array([[1], [1]])
is_c_ind_valid = np.array([[True]])
goal_reward = np.array([[10]])
cost = [1, 2]

reward = create_state_reward(world, c_sub, is_c_ind_valid, goal_reward, cost)
print('Python Reward:')
print(reward)