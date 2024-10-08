import numpy as np
import scipy.io

def equals(a, verbose=False):
    # Load the .mat file
    mat = scipy.io.loadmat("/home/aby/service/alist/guest_init/matrix.mat")
    
    # Extract the matrix from the loaded .mat file
    key = ""
    for k in mat.keys():
        if k not in ['__header__', '__version__', '__globals__']:
            key = k
    
    # Extract matrix and ensure it's a NumPy array
    M_matlab = np.array(mat[key])

    if M_matlab.shape != a.shape:
        if len(a.shape) == 1 and a.shape[0] == M_matlab.shape[1]:
            a_new = a[np.newaxis, :]
            print(f"Adjusted dimension for input matrix from {a.shape} to {a_new.shape}", end=" --> ")
            return equals(a_new, verbose)
        print(f"Wrong dimension, input: {a.shape}; matlab: {M_matlab.shape}")
        return False

    # Use np.allclose with equal_nan=True to compare arrays with NaNs
    comparison = np.allclose(a, M_matlab, equal_nan=True)
    print(comparison)
    if not comparison:
        difference = a - M_matlab
        mse = np.mean(np.square(difference))
        if verbose:
            print(f"The Mean Squared Error between Two Matrices is {mse}")
            
            # Find the indices where the matrices differ
            differing_indices = np.argwhere(~np.isclose(a, M_matlab, equal_nan=True))
            print(f"The matrices differ at the following indices:\n{differing_indices}")
    return comparison
