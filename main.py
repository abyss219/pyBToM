from model import create_belief_state
import torch



def test_create_belief_state():
    # Define a valid indices tensor and grid size
    is_c_ind_valid = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
    n_grid = 1

    # Call the function
    s_sub, s_dim, b_sub, b_to_g, g_to_b = create_belief_state(is_c_ind_valid, n_grid)

    # Output the results
    print("State Subgrid (s_sub):")
    print(s_sub)
    print("\nState Dimensions (s_dim):")
    print(s_dim)
    print("\nBelief Subgrid (b_sub):")
    print(b_sub)
    print("\nMapping from Belief to Grid (b_to_g):")
    print(b_to_g)
    print("\nMapping from Grid to Belief (g_to_b):")
    print(g_to_b)

test_create_belief_state()