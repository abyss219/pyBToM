from model import barycentric_coord
import torch



# Test Case 1
b_sub = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
b_to_g = torch.tensor([[2, 0], [0, 3]], dtype=torch.float32)

neighbor_g_ind, b_coord = barycentric_coord(b_sub, b_to_g)
print("Test Case 1:")
print("neighbor_g_ind:")
print(neighbor_g_ind)
print("b_coord:")
print(b_coord)

# Test Case 2
b_sub = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.float32)
b_to_g = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

neighbor_g_ind, b_coord = barycentric_coord(b_sub, b_to_g)
print("Test Case 2:")
print("neighbor_g_ind:")
print(neighbor_g_ind)
print("b_coord:")
print(b_coord)