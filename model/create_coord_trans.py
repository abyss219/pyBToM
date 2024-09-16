import torch
from .create_coord_space import create_coord_space
from .sub2indv import sub2indv


def create_coord_trans(world, action, p_action_fail):
    """
    Create deterministic coordinate transition matrix with absorbing end state.

    Input:
        - world: a list of world objects
        - action: PyTorch tensor of actions, shape (2, n_action)
        - p_action_fail: probability of action failure (float)

    Output:
        - c_trans: PyTorch tensor of size (n_c_ind, n_c_ind, n_world, n_action). 
                   c_trans[cj, ci, w, a] = p(cj | ci, a, w).
        - c_sub: PyTorch tensor of coordinate subscripts
        - is_c_ind_valid: PyTorch boolean tensor indicating valid indices
    """

    n_action = action.shape[1]
    
    # Create coordinate space
    c_sub, is_c_ind_valid = create_coord_space(world)
    n_c_sub = c_sub.shape[1]
    n_world, n_c_ind = is_c_ind_valid.shape
    
    # Initialize transition matrix
    c_trans = torch.zeros((n_c_ind, n_c_ind, n_world, n_action), dtype=torch.float32)
    
    absorbing_state = n_c_ind - 1  # Explicit index for the absorbing state

    for nw in range(n_world):
        graph_sz = torch.tensor(world[nw]['graph_sz'], dtype=torch.int64)
    
        c_sub_valid = torch.nonzero(is_c_ind_valid[nw, :n_c_sub], as_tuple=False).view(-1)
        c_sub_invalid = torch.nonzero(~is_c_ind_valid[nw, :n_c_sub], as_tuple=False).view(-1)
    
        in_bounds = graph_sz.unsqueeze(1).repeat(1, c_sub_valid.numel())
    
        # Initialize is_goal_ind
        is_goal_ind = torch.zeros(n_c_sub, dtype=torch.bool)
    
        goal_pose = [gp for gp in world[nw]['goal_pose'] if len(gp) > 0]
        goal_pose = torch.tensor(goal_pose, dtype=torch.int64)
        if goal_pose.dim() == 1:
            goal_pose = goal_pose.unsqueeze(1)
        if c_sub.shape[1] > 0 and goal_pose.numel() > 0:
            is_goal_ind = (goal_pose.unsqueeze(2) == c_sub.unsqueeze(1)).all(dim=0).any(dim=1)
    
        for na in range(n_action - 1):
            # Moves from invalid c_subs
            if c_sub_invalid.numel() > 0:
                c_trans[:, c_sub_invalid, nw, na] = float('nan')
    
            # Compute move_sub
            move_sub = c_sub[:, c_sub_valid] + action[:, na].unsqueeze(1).to(torch.int64)
            move_in_bounds_mask = (move_sub <= in_bounds).all(dim=0) & (move_sub > 0).all(dim=0)
            move_in_bounds = torch.nonzero(move_in_bounds_mask, as_tuple=False).view(-1)
            move_sub_in_bounds = move_sub[:, move_in_bounds]
    
            # Proceed only if there are valid moves
            if move_sub_in_bounds.numel() > 0:
                move_ind = sub2indv(graph_sz.tolist(), move_sub_in_bounds)
                move_ind = move_ind.to(torch.int64)
                move_valid = is_c_ind_valid[nw, move_ind]
                n_move_valid = move_valid.sum().item()
    
                if n_move_valid > 0:
                    # Valid moves
                    move_ind_valid = move_ind[move_valid]
                    c_sub_valid_indices = c_sub_valid[move_in_bounds[move_valid]]
    
                    c_trans_sub = torch.zeros((4, n_move_valid), dtype=torch.int64)
                    c_trans_sub[0, :] = move_ind_valid  # cj
                    c_trans_sub[1, :] = c_sub_valid_indices  # ci
                    c_trans_sub[2, :] = nw  # nw
                    c_trans_sub[3, :] = na  # na
    
                    # Valid moves succeed
                    c_trans[c_trans_sub[0], c_trans_sub[1], c_trans_sub[2], c_trans_sub[3]] = 1 - p_action_fail
    
                    # Valid moves fail
                    c_trans_sub_fail = c_trans_sub.clone()
                    c_trans_sub_fail[0, :] = c_trans_sub_fail[1, :]  # cj = ci
                    c_trans[c_trans_sub_fail[0], c_trans_sub_fail[1], c_trans_sub_fail[2], c_trans_sub_fail[3]] += p_action_fail
    
            # Invalid moves: moves out of bounds or into obstacles
            c_trans_stay = is_c_ind_valid[nw, :n_c_sub].clone()
            if move_in_bounds.numel() > 0 and move_valid.numel() > 0:
                indices_to_set_false = c_sub_valid[move_in_bounds[move_valid]]
                c_trans_stay[indices_to_set_false] = False
            c_trans_stay_ind = torch.nonzero(c_trans_stay, as_tuple=False).view(-1)
    
            if c_trans_stay_ind.numel() > 0:
                c_trans_sub = torch.zeros((4, c_trans_stay_ind.numel()), dtype=torch.int64)
                c_trans_sub[0, :] = c_trans_sub[1, :] = c_trans_stay_ind  # cj = ci = stay indices
                c_trans_sub[2, :] = nw
                c_trans_sub[3, :] = na
                c_trans[c_trans_sub[0], c_trans_sub[1], c_trans_sub[2], c_trans_sub[3]] = 1
            
        # Set absorbing state transition
        c_trans[absorbing_state, absorbing_state, nw, :] = 1
    
        goal_indices = torch.nonzero(is_goal_ind, as_tuple=False).view(-1)
        indices = torch.cat((goal_indices, torch.tensor([absorbing_state], dtype=torch.int64)))
        if indices.numel() > 0:
            c_trans[absorbing_state, indices, nw, n_action - 1] = 1
    
        not_goal_indices = torch.nonzero(~is_goal_ind, as_tuple=False).view(-1)
        n_not_goal = not_goal_indices.numel()
        if n_not_goal > 0:
            c_trans[not_goal_indices[:, None], not_goal_indices[None, :], nw, n_action - 1] = torch.eye(n_not_goal, dtype=c_trans.dtype)
    
    return c_trans, c_sub, is_c_ind_valid
