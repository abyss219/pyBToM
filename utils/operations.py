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
    linear_indices = np.ravel_multi_index(result, arr.shape, order='F')
    
    return linear_indices

def length(a:np.ndarray):
    """
    Mimics MATLAB's length function.
    Returns the size of the largest dimension of the input array.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)


    return max(a.shape)



def pad_sublists(input_list, padding_value=np.nan):
    """
    Pads sublists to make all sublists the same length.
    
    Parameters:
    - input_list: list of lists (with potentially different lengths)
    - padding_value: value to pad the shorter sublists (default: np.nan)
    
    Returns:
    - A list where all sublists are of the same length.
    """
    # Find the length of the longest sublist
    max_len = max(len(sublist) for sublist in input_list)
    
    # Pad each sublist to the maximum length with the padding_value
    padded_list = [sublist + [padding_value] * (max_len - len(sublist)) for sublist in input_list]
    
    return padded_list


def repmat(A, reps):
    """
    Simulates the MATLAB repmat function in Python using NumPy.
    
    Parameters:
    A (ndarray): Input array to be replicated.
    reps (tuple or list): A list or tuple specifying the replication factors for each dimension.
    
    Returns:
    ndarray: The replicated array.
    """
    # Ensure the input array A is a NumPy array
    A = np.array(A)

    # Convert reps to a tuple if it's not already
    if not isinstance(reps, tuple):
        reps = tuple(reps)

    # Check the number of dimensions in A
    A_ndim = A.ndim
    reps_len = len(reps)

    # If A has fewer dimensions than reps, pad A with singleton dimensions
    if A_ndim < reps_len:
        A = np.reshape(A, A.shape + (1,) * (reps_len - A_ndim))

    # Now A has the same number of dimensions as reps
    # Use numpy's tile function to replicate A according to reps
    return np.tile(A, reps)

def reshape(A, reps):
    return np.reshape(A, reps, order='F')