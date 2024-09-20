import torch
from .barycentric_coord import barycentric_coord
from .ind2subv import ind2subv
from utils import equals

def create_belief_state_sptrans(s_dim, s_sub, w_trans, c_trans, obs_dist, b_sub, b_to_g, g_ind_to_b_ind):
    # Inputs are PyTorch tensors
    # s_dim: tuple or list with dimensions (n_b_sub, n_c_sub)
    # All indices are adjusted for zero-based indexing in Python
    
    
    a = w_trans
    print(a.size())
    print(torch.sum(a))
    print(torch.nonzero(a))
    print(equals(a, "w_trans"))
    return

    n_b_sub, n_c_sub = s_dim[0], s_dim[1]

    n_world = w_trans.size(0)
    n_action = c_trans.size(3)
    n_obs = obs_dist.size(0)


    n_w_next = (w_trans > 0).sum(dim=0)
    max_w_next = n_w_next.max().item()

    n_c_next = (c_trans > 0).sum(dim=0)
    max_c_next = n_c_next.max().item()

    n_obs_next = (obs_dist > 0).sum(dim=0)
    max_obs_next = n_obs_next.max().item()

    max_co_ind = n_world * max_w_next * max_c_next * max_obs_next
    

    # wco_trans represents p(w2, [c2,o]_ind | w1, c1, a)
    wco_trans = torch.zeros(n_world, max_co_ind, n_world, n_c_sub, n_action)

    # co_trans_ind gives possible [c2,o]_inds given c1, a (for all wi)
    co_trans_ind = torch.zeros(max_co_ind, n_c_sub, n_action, dtype=torch.int32)

    # Precompute to use below
    obs_dist_perm = obs_dist.permute(2, 1, 0).unsqueeze(3).expand(-1, -1, -1, n_world)
    
    
    # obs_dist_perm shape: [1, n_c_sub, n_obs, n_world]

    w_trans_precomp = w_trans.reshape(n_world, 1, n_world).expand(-1, n_c_sub, -1)
    # w_trans_precomp shape: [n_world, n_c_sub, n_world]

    n_ca_ind = n_c_sub * n_action
    
    ca_sub = ind2subv([n_c_sub, n_action], torch.arange(1, n_ca_ind + 1)).T
    # ca_sub shape: [2, n_ca_ind]
    
    max_co_ind2 = 0
    
    
    for ni in range(n_ca_ind):
        ci = ca_sub[0, ni] - 1
        ai = ca_sub[1, ni] - 1

        temp = c_trans[:, ci, :, ai].squeeze().t()  

        wc_pred = temp.unsqueeze(-1).repeat(1, 1, n_world) * w_trans_precomp 
        
        wco_pred = obs_dist_perm * wc_pred.reshape(n_world, n_c_sub, 1, n_world).repeat(1, 1, n_obs, 1)
        wco_pred = wco_pred.reshape(n_world, n_c_sub * n_obs, n_world)

        # Find indices where wco_pred > 0
        co_ind_tmp = torch.nonzero((wco_pred > 0).any(dim=2).any(dim=0))

                
        n_co_ind_tmp = co_ind_tmp.numel()
        
        
        if n_co_ind_tmp > 0:

            wco_trans[:, :n_co_ind_tmp, :, ci, ai] = wco_pred[:, co_ind_tmp, :]
            # print(torch.sum(wco_trans))
            
            co_trans_ind[:n_co_ind_tmp, ci, ai] = co_ind_tmp
            if n_co_ind_tmp > max_co_ind2:
                max_co_ind2 = n_co_ind_tmp


    
    return
    # Compress these to save memory
    wco_trans = wco_trans[:, :max_co_ind2, :, :, :]
    co_trans_ind = co_trans_ind[:max_co_ind2, :, :]
    print(max_co_ind2)
    print(torch.sum(wco_trans))
    return
    n_bc_ind = n_b_sub * n_c_sub
    max_s_trans_ind = max_obs_next * n_world * max_c_next
    s_trans = torch.zeros(max_s_trans_ind, n_bc_ind, n_action)
    s_trans_ind = torch.zeros(max_s_trans_ind, n_bc_ind, n_action, dtype=torch.int32)
    # Compute s_trans and s_trans_ind
    for bi in range(n_b_sub):
        b_sub_bi = b_sub[:, bi].reshape(1, 1, n_world)
        

        b_sub_bi = b_sub_bi.unsqueeze(-1).unsqueeze(-1)  # Now shape is [1, 1, 3, 1, 1]
        b_sub_bi = b_sub_bi.repeat(n_world, max_co_ind2, 1, n_c_sub, n_action)
        
        b2 = (wco_trans * b_sub_bi).sum(dim=2, keepdim=True)
        # b2 shape: [n_world, max_co_ind2, n_c_sub, n_action]
        

        b2_flat = b2.view(n_world, -1)
        b2_sum = (b2 > 0).sum(dim=0)
        
        b2_valid_ind = torch.nonzero(b2_sum > 0).T
        
        b2_valid_sub = ind2subv([max_co_ind2, n_c_sub, n_action],b2_valid_ind).T
        

        
        b2_valid = b2[:, b2_valid_ind]
        b2_valid_sum = b2_valid.sum(dim=0)
        b2_valid_norm = b2_valid / b2_valid_sum

        # Compute barycentric coordinates (assuming barycentric_coord is defined)
        neighbors, lambda_ = barycentric_coord(b2_valid_norm, b_to_g)
        neighbors_valid = (neighbors > 0) & (lambda_ > 0)
        neighbors_valid_ind = torch.nonzero(neighbors_valid)
        n_neighbors_valid_ind = neighbors_valid_ind.size(0)

        # Get previous sa_inds
        ca_valid_ind = b2_valid_ind[neighbors_valid_ind[:, 1]]
        sa_valid_ind = bi * n_c_sub * n_action + ca_valid_ind

        # Get b_inds of all neighbors
        neighbors_valid_b_ind = g_ind_to_b_ind[neighbors[neighbors_valid_ind[:, 0], neighbors_valid_ind[:, 1]]]

        # Get co_inds of all neighbors
        co_valid_ind = co_trans_ind[b2_valid_ind[neighbors_valid_ind[:, 1]], neighbors_valid_ind[:, 1] // n_action, neighbors_valid_ind[:, 1] % n_action]
        co_valid_sub = torch.stack([co_valid_ind // n_obs, co_valid_ind % n_obs], dim=1).t()

        # Compute bc[o]_inds
        bc_valid_ind = neighbors_valid_b_ind * n_c_sub + co_valid_sub[0, :]

        # Update s_trans and s_trans_ind
        left_inds = torch.zeros_like(ca_valid_ind)
        left_inds[torch.cat([torch.tensor([0]), (ca_valid_ind[1:] != ca_valid_ind[:-1]).nonzero().squeeze() + 1])] = 1
        left_inds = left_inds.cumsum(dim=0) - 1

        s_trans_valid_ind = left_inds * n_bc_ind * n_action + sa_valid_ind
        s_trans.view(-1)[s_trans_valid_ind] = b2_valid_sum[neighbors_valid_ind[:, 1]] * lambda_[neighbors_valid_ind[:, 0], neighbors_valid_ind[:, 1]]
        s_trans_ind.view(-1)[s_trans_valid_ind] = bc_valid_ind

    # Replace zero indices with 1 to allow indexing value function with s_trans_ind
    s_trans_ind[s_trans_ind == 0] = 1

    return s_trans, s_trans_ind