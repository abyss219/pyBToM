import numpy as np
import scipy.io

def equals(a):
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
        print(f"Wrong dimension, input: {a.shape}; matlab: {M_matlab.shape}")
        return False

    # Use np.allclose with equal_nan=True to compare arrays with NaNs
    comparison = np.allclose(a, M_matlab, equal_nan=True)
    print(comparison)
    if not comparison:
        difference = a - M_matlab
        mse = np.mean(np.square(difference))
        print(f"The Mean Squared Error between Two Matrices is {mse}")
    return comparison
