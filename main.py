from model import barycentric_coord
import torch
import numpy as np




# Initial tensor of shape [1, 1, 3]
b_sub_bi = torch.tensor([[[1], [2], [3]]])  # Shape [1, 1, 3]

# Define the repeat sizes
n_world = 3          # Example value
max_co_ind2 = 4      # Example value
n_c_sub = 76         # Example value
n_action = 6         # Example value

# Now repeat the tensor along each dimension
# b_sub_bi_repeated = np.tile(b_sub_bi, [n_world, max_co_ind2, 1, n_c_sub, n_action])

# Check the final shape
print(b_sub_bi.shape)  # This should output torch.Size([3, 4, 3, 76, 6])
