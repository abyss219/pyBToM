import torch

def ind2subv(siz, index):
    """
    Convert linear index/indices to subscript indices for a tensor of given size.
    
    Parameters:
    - siz: A list or tensor specifying the size of each dimension of the tensor.
    - index: A tensor containing the linear index or indices to be converted.
    
    Returns:
    - sub: A tensor where each row corresponds to the subscript indices.
    """
    
    # Number of dimensions
    n = len(siz)
    
    # Cumulative product of sizes
    if not isinstance(siz, torch.Tensor):
        cum_size = torch.cumprod(torch.tensor(siz), dim=0)
    else:
        cum_size = siz
    prev_cum_size = torch.cat([torch.tensor([1]), cum_size[:-1]])
    
    # Convert to zero-based index
    index = index - 1
    
    # Prepare for repeated broadcasting
    index = index.view(-1, 1)  # Make sure index is a column vector
    cum_size = cum_size.view(1, -1)  # Row vector
    prev_cum_size = prev_cum_size.view(1, -1)  # Row vector
    
    # Calculate the subscript indices
    sub = index % cum_size
    sub = torch.floor(sub / prev_cum_size).long() + 1
    
    return sub