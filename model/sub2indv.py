import torch

def sub2indv(siz:torch.Tensor, sub):
    """
    Convert subscripts to linear indices.

    Parameters:
    siz : list or tuple of ints
        Size of the array into which sub is an index.
    sub : Tensor of size (n, nsub), dtype int64
        Each column sub[:, i] is the ith set of subscripts into the array.

    Returns:
    ind : Tensor of size (nsub,), dtype int64
        Linear indices into the array of size siz.
    """
    if sub.numel() == 0:
        return torch.empty(0, dtype=torch.int64)
    
    nsub = sub.size(1)
    if not isinstance(siz, torch.Tensor):
        siz_tensor = torch.tensor(siz, dtype=torch.int64)
    else:
        siz_tensor = siz
        
    k = torch.cat((torch.tensor([1], dtype=torch.int64), torch.cumprod(siz_tensor[:-1], 0))).unsqueeze(1)
    sub = sub.to(dtype=torch.int64)
    ind = torch.sum((sub - 1) * k, dim=0) + 1  # Subtract 1 from sub and add 1 to the result
    return ind