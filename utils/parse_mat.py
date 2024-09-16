import scipy.io
import numpy as np

import numpy as np
import torch
import scipy.io

def convert_mat_data(data):
    """
    Recursively converts MATLAB structs to dictionaries and NumPy arrays to lists.

    Parameters:
    - data: The data loaded from a MATLAB .mat file.

    Returns:
    - The converted data with MATLAB structs as dicts and NumPy arrays as lists.
    """
    if isinstance(data, dict):
        # Process each value in the dictionary recursively
        return {key: convert_mat_data(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        if data.dtype.type is np.object_:
            # Handle arrays of MATLAB structs or cell arrays
            return [convert_mat_data(element) for element in data]
        else:
            # Convert NumPy array to list
            return data.tolist()
    elif isinstance(data, scipy.io.matlab.mio5_params.mat_struct):
        # Convert MATLAB struct to dict
        return {field: convert_mat_data(getattr(data, field)) for field in data._fieldnames}
    elif isinstance(data, (np.void,)):
        # Handle MATLAB structs represented as np.void
        return {name: convert_mat_data(data[name]) for name in data.dtype.names}
    else:
        # Return data as is (e.g., numbers, strings)
        return data
