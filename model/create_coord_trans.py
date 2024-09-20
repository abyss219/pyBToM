from .create_coord_space import create_coord_space
from .sub2indv import sub2indv
from utils import equals, find, pad_sublists, length
import numpy as np

def create_coord_trans(world, action, p_action_fail):
    """
    Create deterministic coordinate transition matrix with absorbing end state.

    Input:
      world:
      action:
      p_action_fail:

    Output:
      c_trans: size(n_c_ind,n_c_ind,n_world,n_action). c_trans[cj,ci,a,w] = p(cj|ci,a,w).
      c_sub:
      is_c_ind_valid:
    """

    n_action = action.shape[1]

    c_sub, is_c_ind_valid = create_coord_space(world)


    n_c_sub = c_sub.shape[1]

    n_world,n_c_ind = is_c_ind_valid.shape

    c_trans = np.zeros((n_c_ind, n_c_ind, n_world, n_action))

    for nw in range(n_world):
        graph_sz = np.array(world[nw]['graph_sz'])
        c_sub_valid = find(is_c_ind_valid[nw:nw+1, :n_c_sub])
        

        # Assuming save functionality is not required in Python

        n_c_sub_valid = len(c_sub_valid)

        c_sub_invalid = find(np.logical_not(is_c_ind_valid[nw:nw+1, :]))

        in_bounds = np.tile(graph_sz[:, np.newaxis], (1, n_c_sub_valid))

        is_goal_ind = np.zeros((n_c_sub,), dtype=bool)[:, np.newaxis]

        goal_pose = np.array(pad_sublists(world[nw]['goal_pose']))
        for nc in range(n_c_sub):
            is_goal_ind[nc] = np.any(np.all(goal_pose == c_sub[:, nc], axis=1))
        
        for na in range(n_action - 1):  # MATLAB indices from 1 to n_action - 1
            # Moves from invalid c_subs
            
            c_trans[:, c_sub_invalid, nw, na] = np.nan

            # Compute move_sub
            move_sub = c_sub[:, c_sub_valid] + np.tile(action[:, na:na+1], (1, n_c_sub_valid))

            # Determine in-bounds moves
            condition1 = np.all(move_sub <= in_bounds, axis=0)
            condition2 = np.all(move_sub > 0, axis=0)
            # Combine the two conditions with a logical AND
            result = condition1 & condition2
            move_in_bounds = find(result)[np.newaxis, :]

            # Convert subscripts to linear indices
            move_ind = sub2indv(graph_sz[:, np.newaxis], move_sub[:, move_in_bounds[0]])[np.newaxis, :]

            # Check which moves are valid
            move_valid = is_c_ind_valid[nw, move_ind - 1]

            n_move_valid = np.sum(move_valid)

            # Valid moves
            c_trans_sub = np.zeros((4, n_move_valid), dtype=int)


            c_trans_sub[0, :] = move_ind[move_valid]
            c_trans_sub[1, :] = c_sub_valid[move_in_bounds[move_valid]] + 1

            c_trans_sub[2, :] = nw + 1
            c_trans_sub[3, :] = na + 1

            # Valid moves succeed
            c_trans_indices = sub2indv(c_trans.shape, c_trans_sub)
            c_trans_flat = c_trans.ravel(order='F')
            c_trans_flat[c_trans_indices - 1] = 1 - p_action_fail
            c_trans = c_trans_flat.reshape(c_trans.shape, order='F')

            # Valid moves fail (stay in the same position)
            c_trans_sub[0, :] = c_trans_sub[1, :]
            c_trans_indices = sub2indv(c_trans.shape, c_trans_sub)
            c_trans.flat[c_trans_indices] += p_action_fail

            # Invalid moves (stay in the same position)
            c_trans_stay = is_c_ind_valid[nw, :n_c_sub].copy()
            c_trans_stay[c_sub_valid[move_in_bounds[move_valid]]] = False
            c_trans_stay_ind = find(c_trans_stay)
            n_c_trans_stay_ind = length(c_trans_stay_ind)

            c_trans_sub = np.zeros((4, n_c_trans_stay_ind), dtype=int)
            c_trans_sub[0, :] = c_trans_stay_ind + 1
            c_trans_sub[1, :] = c_trans_stay_ind + 1
            c_trans_sub[2, :] = nw + 1
            c_trans_sub[3, :] = na + 1

            c_trans_indices = sub2indv(c_trans.shape, c_trans_sub)
            

            c_trans_flat = c_trans.ravel(order='F')
            c_trans_flat[c_trans_indices - 1] = 1
            c_trans = c_trans_flat.reshape(c_trans.shape, order='F')


        c_trans[-1, -1, nw, :] = 1
        g = np.append(is_goal_ind.squeeze(), 1)

        c_trans[-1,g,nw, n_action - 1] = 1


        g = np.append(is_goal_ind, 1)
        g = np.logical_not(g)


        not_goal_count = np.sum(np.logical_not(is_goal_ind))  # Get the number of non-goal indices
        identity_matrix = np.eye(not_goal_count)

        c_trans[np.ix_(g, g, [nw], [n_action - 1])] = identity_matrix.reshape(not_goal_count, not_goal_count, 1, 1)


    return c_trans, c_sub, is_c_ind_valid

