from .barycentric_coord import barycentric_coord
from .ind2subv import ind2subv
from .sub2indv import sub2indv
from utils import repmat, reshape, find, length

import numpy as np

def create_belief_state_sptrans(s_dim, s_sub, w_trans, c_trans, obs_dist:np.ndarray, b_sub, b_to_g, g_ind_to_b_ind):
    # [s_trans, s_trans_ind] = create_belief_state_sptrans(s_dim, s_sub, w_trans, c_trans, obs_dist, b_sub, b_to_g, g_ind_to_b_ind)
    #
    # Input:
    #   s_dim:
    #   w_trans:
    #   c_trans:
    #   obs_dist:
    #   b_sub:
    #   b_to_g:
    #   g_ind_to_b_ind:
    #
    # Output:
    #   s_trans[i,j,k] = p(s_trans_ind i | from s_ind j given action k)
    #     s_trans = size(n_s_trans_ind, n_s_ind, n_action)
    #   s_trans_ind[i,j] = next belief state | trans i from s_ind j
    #     s_trans_ind = size(n_s_trans_ind, n_s_ind, n_action)
    #
    # TODO:
    #   -currently s_trans_inds is unordered, may contain duplicates, items don't occupy adjacent inds

    n_b_sub, n_c_sub = s_dim[0], s_dim[1]

    n_world = w_trans.shape[0]
    n_action = c_trans.shape[3]
    n_obs = obs_dist.shape[0]

    n_w_next = np.sum(w_trans > 0, axis=1)
    max_w_next = np.max(n_w_next)

    n_c_next = np.sum(c_trans > 0, axis=0)
    max_c_next = np.max(n_c_next)

    n_obs_next = np.sum(obs_dist > 0, axis=0)
    max_obs_next = np.max(n_obs_next)

    max_co_ind = n_world * max_w_next * max_c_next * max_obs_next

    # wco_trans represents p(w2,[c2,o]_ind | w1, c1, a)
    wco_trans = np.zeros((n_world, max_co_ind, n_world, n_c_sub, n_action))

    # co_trans_ind gives possible [c2,o]_inds given c1, a (for all wi)
    co_trans_ind = np.zeros((max_co_ind, n_c_sub, n_action), dtype='uint32')

    # CB: precompute to use below
    
    obs_dist_perm = repmat(obs_dist.transpose(2, 1, 0), (1, 1, 1, n_world))
    w_trans_precomp = repmat(w_trans.reshape(n_world, 1, n_world), (1, n_c_sub, 1))

    n_ca_ind = n_c_sub * n_action
    ca_indices = np.arange(1, n_ca_ind + 1)
    ca_sub = ind2subv([n_c_sub, n_action], ca_indices).T

    # compute wco_trans, wco_trans_ind: wca -> wco_ind
    max_co_ind2 = 0

    for ni in range(n_ca_ind):
        ci, ai = ca_sub[:, ni]
        ci -= 1  # Adjust for zero-based indexing
        ai -= 1

        wc_pred = repmat(np.squeeze(c_trans[:,ci:ci+1,:,ai:ai+1]).T,[1, 1, n_world]) * w_trans_precomp
        wco_pred = obs_dist_perm * repmat(reshape(wc_pred,[n_world, n_c_sub, 1, n_world]),[1, 1, n_obs, 1])
        
        wco_pred = reshape(wco_pred, [n_world, n_c_sub * n_obs, n_world])


        co_ind_tmp = find(np.any(np.any(wco_pred > 0, axis=2), axis=0))
        n_co_ind_tmp = length(co_ind_tmp)

        if n_co_ind_tmp > 0:
            wco_trans[:, :n_co_ind_tmp, :, ci, ai] = wco_pred[:, co_ind_tmp, :]
            co_trans_ind[:n_co_ind_tmp, ci, ai] = co_ind_tmp + 1  # Adjust for one-based indexing
                
            if n_co_ind_tmp > max_co_ind2:
                max_co_ind2 = n_co_ind_tmp


    # compress these to save memory
    wco_trans = wco_trans[:, 0:max_co_ind2, :, :, :]
    co_trans_ind = co_trans_ind[0:max_co_ind2, :, :]

    n_bc_ind = n_b_sub * n_c_sub
    max_s_trans_ind = max_obs_next * n_world * max_c_next
    s_trans = np.zeros((max_s_trans_ind, n_bc_ind, n_action))
    s_trans_ind = np.zeros((max_s_trans_ind, n_bc_ind, n_action), dtype='uint32')

    # compute s_trans, s_trans_ind

    for bi in range(n_b_sub):
        b2 = np.sum(wco_trans * repmat(reshape(b_sub[:,bi:bi+1],[1, 1, n_world]),[n_world, max_co_ind2, 1, n_c_sub, n_action]),axis=2, keepdims=True)

        # b2_valid_sub: [co_ind_next; c_ind_prev; a_ind_prev] for each b2_valid_ind
        b2_valid_ind = find(np.sum(reshape(b2, (b2.shape[0], -1)) > 0, axis=0, keepdims=True))
        b2_valid_sub = ind2subv([max_co_ind2, n_c_sub, n_action], b2_valid_ind + 1).T

        b2_valid = reshape(b2, (b2.shape[0], -1))
        b2_valid = b2_valid[:, b2_valid_ind]

        b2_valid_sum = np.sum(b2_valid, axis=0, keepdims=True)
        b2_valid_norm = b2_valid / repmat(b2_valid_sum, [n_world, 1])


        # CB: b2_valid_norm_unique? apply barycentric_coord1 to as few b_subs as possible...
        neighbors, lambda_ = barycentric_coord(b2_valid_norm, b_to_g)
        neighbors_valid = (neighbors > 0) & (lambda_ > 0)
        neighbors_valid_ind = find(neighbors_valid.T)

        n_neighbors_valid_ind = len(neighbors_valid_ind)


        # [neighbor_ind; b2_valid_ind]
        neighbors_valid_sub = ind2subv(neighbors.shape, neighbors_valid_ind + 1).T

        tmp = b2_valid_sub[np.array([1, 2]), :][:, neighbors_valid_sub[1:2, :].squeeze() - 1]
    


        # get previous sa_inds
        ca_valid_ind = sub2indv([n_c_sub, n_action], tmp)
        sa_valid_ind = sub2indv([n_b_sub, n_c_sub * n_action], np.vstack([repmat(bi + 1, [1, n_neighbors_valid_ind]), ca_valid_ind]))
        
        # get next bc_inds

        # get b_inds of all neighbors
        neighbors_flattened = neighbors.ravel(order='F')  # Flatten in column-major order
        neighbors_valid_t = neighbors_flattened[neighbors_valid_ind]  # Get the valid neighbors and reshape
        neighbors_valid_b_ind = g_ind_to_b_ind[neighbors_valid_t.astype(int) - 1]

            

        

        # get co_inds of all neighbors
        co_trans_ind_flattened = co_trans_ind.ravel(order='F')
        co_valid_ind = co_trans_ind_flattened[b2_valid_ind[neighbors_valid_sub[1:2, :] - 1]].reshape(-1, 1).T
        co_valid_sub = ind2subv([n_c_sub, n_obs],co_valid_ind).T
        bc_valid_ind = sub2indv([n_b_sub, n_c_sub], np.vstack([neighbors_valid_b_ind[np.newaxis, :] + 1, co_valid_sub[0:1, :]]))[np.newaxis, :]
        ca_valid_ind_diff = find(np.diff(ca_valid_ind))


        left_inds = np.zeros(ca_valid_ind.shape)
        
        
        
        left_inds[np.concatenate([ca_valid_ind_diff, [len(left_inds)-1]])] = np.diff(np.concatenate([[0], ca_valid_ind_diff + 1, [length(ca_valid_ind)]]))
        
        left_inds = left_inds + np.arange(1, len(ca_valid_ind) + 1) - np.cumsum(left_inds)

        s_trans_valid_ind = sub2indv([max_s_trans_ind, n_bc_ind * n_action], np.vstack([left_inds, sa_valid_ind]))

        l_flatten = lambda_.ravel(order='F')
        l = l_flatten[neighbors_valid_ind].reshape(-1, 1, order='F')
        
        s_trans_flatten = s_trans.ravel(order='F')
        s_trans_flatten[s_trans_valid_ind - 1] = b2_valid_sum.squeeze()[neighbors_valid_sub[1, :] - 1] * l.T
        s_trans = s_trans_flatten.reshape(s_trans.shape, order='F')




        s_trans_ind_flatten = s_trans_ind.ravel(order='F')
        s_trans_ind_flatten[s_trans_valid_ind - 1] = bc_valid_ind - 1
        s_trans_ind = s_trans_ind_flatten.reshape(s_trans_ind.shape, order='F')


    s_trans_ind[s_trans_ind == 0] = 1

    return s_trans, s_trans_ind