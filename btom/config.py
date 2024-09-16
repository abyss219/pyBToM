import torch

# [stay, west, east, south, north, eat]
action = torch.tensor([[0, -1,  1,  0,  0,  0],
                       [0,  0,  0, -1,  1,  0]], dtype=torch.int64)
n_action = action.shape[1]

# Grid and belief space settings
n_reward_grid = 7
n_belief_grid = 6
b_sub_space = 0.075
b_precision = 2**16

# View border
view_border = [0.01, 0.01]

# MDP options for planning (fixed)
mdp_options = {
    'stochastic_value': False,
    'stochastic_policy': False,
    'beta': float('inf'),
    'discount': 0.99,
    'max_iter': 50,
    'err_tol': 0.0,
    'sptrans': True,
    'verbose': False
}

# Other parameters
visilibity = False
move_noise = 0.0
obs_noise = 0
move_cost = 1
action_cost = [0] + [move_cost] * (n_action - 1)
# beta_score = 2.5  # This can be set dynamically if needed
