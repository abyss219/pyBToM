import torch
from .sub2indv import sub2indv

def g_sub_to_g_ind(g_sub, dim, k):
    """
    Converts subscript indices to linear indices for an array of size (k+1)^dim.

    Parameters:
    - g_sub (torch.Tensor): Tensor of shape [dim, N], where each column represents subscript indices.
    - dim (int): The number of dimensions.
    - k (int): The maximum index in each dimension (indices range from 0 to k).

    Returns:
    - g_ind (torch.Tensor): Tensor of shape [N], containing the linear indices.
    """
    N = g_sub.shape[1]
    g_ind = torch.zeros(N, dtype=torch.long)

    # Check which columns have all indices within the valid range [0, k]
    g_ind_in_range = torch.all((g_sub <= k) & (g_sub >= 0), dim=0)

    # Get indices of valid columns
    valid_indices = torch.nonzero(g_ind_in_range).squeeze()

    if valid_indices.numel() > 0:
        # Prepare subscripts for valid indices (adding 1 for 1-based indexing)
        subs = g_sub[:, valid_indices] + 1  # MATLAB uses 1-based indexing

        # Create size vector
        siz = torch.tensor([k + 1] * dim, dtype=torch.long)

        # Compute linear indices using sub2indv
        linear_indices = sub2indv(siz, subs)

        # Assign computed indices to the result
        g_ind[valid_indices] = linear_indices

    g_ind_matrix = g_ind.unsqueeze(0)

    return g_ind_matrix

# Test cases
if __name__ == "__main__":
    # Test case 1
    g_sub = torch.tensor([
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2]
    ])
    dim = 3
    k = 2
    g_ind = g_sub_to_g_ind(g_sub, dim, k)
    print("Test case 1 output:", g_ind)
    # Expected output: tensor([ 0, 13, 26])

    # Test case 2 - with out-of-range indices
    g_sub = torch.tensor([
        [0, 1, 2, 3],
        [0, 1, 2, 1],
        [0, 1, 2, 1]
    ])
    dim = 3
    k = 2
    g_ind = g_sub_to_g_ind(g_sub, dim, k)
    print("Test case 2 output:", g_ind)
    # Expected output: tensor([ 0, 13, 26,  0]) - last index is zero due to invalid subscripts

    # Test case 3 - negative indices
    g_sub = torch.tensor([
        [0, 1, -1],
        [0, 1, 1],
        [0, 1, 1]
    ])
    dim = 3
    k = 2
    g_ind = g_sub_to_g_ind(g_sub, dim, k)
    print("Test case 3 output:", g_ind)
    # Expected output: tensor([ 0, 13,  0]) - last index is zero due to negative subscript

    # Test case 4 - larger k value
    g_sub = torch.tensor([
        [2, 3, 4],
        [1, 2, 3],
        [0, 1, 2]
    ])
    dim = 3
    k = 4
    g_ind = g_sub_to_g_ind
