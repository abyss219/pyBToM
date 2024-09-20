import numpy as np

def ind2subv(siz, index):
    """
    ind2subv(siz, index) returns a vector of the equivalent subscript values 
    corresponding to a single index into an array of size siz.
    If index is a vector, then the result is a matrix, with subscript vectors
    as rows.

    Note: This function assumes 1-based indexing, similar to MATLAB.
    
    Parameters:
    - siz: list or array-like, size of each dimension
    - index: integer or array-like of integers, linear indices
    
    Returns:
    - sub: numpy array of subscripts, shape (num_indices, num_dimensions)
    """
    siz = np.array(siz, dtype=np.uint32)
    n = len(siz)
    cum_size = np.cumprod(siz)
    prev_cum_size = np.concatenate(([1], cum_size[:-1]))
    index = np.asarray(index).flatten() - 1  # Adjust for 1-based indexing
    index = index.astype(np.uint32)
    num_indices = index.shape[0]
    sub = np.empty((num_indices, n), dtype=np.uint32)

    for i in range(n):
        sub[:, i] = (index % cum_size[i]) // prev_cum_size[i] + 1  # Compute subscripts

    return sub