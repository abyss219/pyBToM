from model import regular_simplex
import torch
import numpy as np
from utils import equals

dim = 3
k = 2
G_sub, G_ind = regular_simplex(dim, k)
print('G_sub:')
print(G_sub)
print('G_ind:')
print(G_ind)