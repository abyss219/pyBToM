import numpy as np

def sub2indv(siz, sub):
    """
    ind = sub2indv(siz, sub)
    
    Improved version of sub2ind.
    
    INPUTS
        siz - size of array into which sub is an index
        sub - sub[:, i] is the ith set of subscripts into the array.
    
    OUTPUTS
        ind - linear index (or vector of indices) into given array.
    """
    if sub.size == 0:
        return np.array([], dtype=np.uint32)
    
    n = len(siz)
    nsub = sub.shape[1]  # Number of subscript sets

    k = np.array([1] + list(np.cumprod(siz[:-1])), dtype=np.uint32).reshape(n, 1)
    ind = np.sum((sub - 1).astype(np.uint32) * k, axis=0) + 1

    return ind