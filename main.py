from model import barycentric_coord
import torch
import numpy as np
from utils import equals



# Initial tensor of shape [1, 1, 3]
b_sub_bi = torch.tensor([[[1], [2], [3]]])  # Shape [1, 1, 3]

equals(b_sub_bi)

array = [3, 4, 5]

print(array[0:-1])