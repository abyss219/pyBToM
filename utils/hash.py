import torch
import scipy.io

def equals(a):
    mat = scipy.io.loadmat("matrix.mat")
    key = ""
    for k in mat.keys():
        if k not in ['__header__', '__version__', '__globals__']:
            key = k
    M_matlab = torch.tensor(mat[key])  # Convert to PyTorch tensor
    comparison = torch.equal(a, M_matlab)  # Compare element-wise
    print(comparison)
    return comparison