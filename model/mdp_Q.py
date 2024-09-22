import numpy as np

def mdp_Q_policy_to_V(Q, policy):
    # [Q] = Q_policy_to_V1(Q, policy)
    #
    
    dims = Q.ndim
    sz = Q.shape
    Q[np.isnan(Q)] = 0
    # policy at invalid poses is NaN
    V = np.sum(Q * policy, axis=dims - 1)
    return V, policy


def mdp_Q_to_policy(Q, beta, stochastic_policy=1):
    # [policy] = mdp_Q_to_policy1(Q, beta, stochastic_policy)
    #

    dims = Q.ndim
    sz = Q.shape

    if stochastic_policy:
        # Compute max_Q along the last dimension
        max_Q = np.max(Q, axis=dims - 1, keepdims=True)
        # Tile max_Q to match the dimensions of Q
        tile_dims = [1] * dims
        tile_dims[-1] = sz[-1]
        max_Q_tiled = np.tile(max_Q, tile_dims)
        policy = np.exp(beta * (Q - max_Q_tiled))
        policy[np.isnan(policy)] = 0
    else:
        policy = np.zeros_like(Q)
        max_Q = np.max(Q, axis=dims - 1, keepdims=True)
        tile_dims = [1] * dims
        tile_dims[-1] = sz[-1]
        max_Q_tiled = np.tile(max_Q, tile_dims)
        policy = (Q == max_Q_tiled).astype(float)

    sum_policy = np.sum(policy, axis=dims - 1, keepdims=True)
    sum_policy_tiled = np.tile(sum_policy, tile_dims)
    with np.errstate(divide='ignore', invalid='ignore'):
        policy = policy / sum_policy_tiled

    return policy


def mdp_Q_VI(V, trans, reward, options):
    # [Q, V, n_iter, err] = mdp_Q_VI(V, trans, reward, options)
    #
    # Run value iteration on an MDP
    #
    # Input:
    #  - V: None or shape (n_state,)
    #  - trans: shape (n_state, n_state, n_action)
    #    - trans[i, j, k] = p(s'=i | s=j, a=k)
    #  - reward: shape (n_state, n_action), the reward for each state-action pair.
    #  - options: dictionary with keys
    #    'stochastic_value', 'beta', 'discount', 'max_iter',
    #    'err_tol', 'sptrans', 'trans_ind', 'verbose'
    #
    # Output:
    #  - Q: shape (n_state, n_action)
    #  - V: shape (n_state,)
    #  - n_iter: number of iterations
    #  - err: array of errors over iterations

    n_state, n_action = reward.shape

    if len(V) == 0:
        V = np.zeros(n_state)
    elif not len(V) == n_state:
        raise ValueError('mdp_Q_VI() - V')

    Q = np.zeros((n_state, n_action))

    stochastic_value = options['stochastic_value']
    beta = options['beta']
    discount = options['discount']
    max_iter = options['max_iter']
    err_tol = options['err_tol']
    sptrans = options['sptrans']
    verbose = options['verbose']

    err = np.zeros(max_iter)
    V = np.array(V)

    

    for n_iter in range(1, max_iter + 1):
        if sptrans:
            # Q = discount * squeeze(sum(V(options.trans_ind) .* trans, 1)) + reward;
            V_trans_ind = V[options['trans_ind'] - 1]  # Assuming options['trans_ind'] is provided
            temp = V_trans_ind * trans  # Element-wise multiplication
            Q = discount * np.squeeze(np.sum(temp, axis=0)) + reward
        else:
            state_valid = ~np.isnan(V)
            for na in range(n_action):
                trans_valid = trans[state_valid][:, :, na]
                EV = trans_valid.T @ V[state_valid]
                Q[:, na] = discount * EV + reward[:, na]

        policy = mdp_Q_to_policy(Q, beta, stochastic_value)

        V_new, _ = mdp_Q_policy_to_V(Q, policy)

        err[n_iter - 1] = np.max(np.abs(V - V_new))
        converge = err[n_iter - 1] < err_tol

        V = V_new

        if verbose:
            print(f'mdp_Q_VI() -- iter {n_iter}/{max_iter} completed')

        if converge:
            break

    err = err[:n_iter]

    return Q, V, n_iter, err