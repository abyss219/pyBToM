import numpy as np

def find(arr, condition=None):
    """
    Mimic the behavior of MATLAB's find function for a multi-dimensional NumPy array.
    
    Parameters:
    arr (numpy.ndarray): The input array.
    condition (function, optional): A condition function that returns a boolean mask.
                                    If None, finds non-zero elements.
    
    Returns:
    numpy.ndarray: Linear indices of elements that satisfy the condition.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.ndarray(arr)

    if condition is None:
        # If no condition is provided, return the indices of non-zero elements
        result = np.nonzero(arr)
    else:
        # Apply the condition to the array and return indices of elements that satisfy the condition
        result = np.nonzero(condition(arr))
    
    # Convert multi-dimensional indices to linear indices
    linear_indices = np.ravel_multi_index(result, arr.shape)
    
    return linear_indices





siz = (3, 4, 5)
sub = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])  # Example subscripts
ind = sub2indv(siz, sub)
print(ind)