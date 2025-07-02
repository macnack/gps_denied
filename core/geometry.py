import torch
from .sampling import grid_bilinear_sampling

def H_to_param(H: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 3x3 homographies to 8D parameter vectors by subtracting identity
    and flattening the top 8 elements (ignores scale).

    Args:
        H (Tensor): [N, 3, 3] batch of homography matrices

    Returns:
        Tensor: [N, 8, 1] parameter vectors
    """
    batch_size = H.shape[0]
    device = H.device

    # Identity matrix, repeated for batch
    I = torch.eye(3, device=device).expand(batch_size, 3, 3)

    # Subtract identity to get delta from identity
    p = H - I

    # Flatten and keep first 8 parameters (drop H[2,2])
    p = p.reshape(batch_size, 9, 1)
    p = p[:, 0:8, :]

    return p


def param_to_H(p: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 8-parameter vectors to 3x3 homography matrices by adding identity.

    Args:
        p: Tensor of shape [N, 8, 1]

    Returns:
        H: Tensor of shape [N, 3, 3]
    """
    batch_size = p.size(0)
    device = p.device

    # Add last (scale) element to get 9 params
    z = torch.zeros(batch_size, 1, 1, device=device)
    p_ = torch.cat((p, z), dim=1)  # shape: [N, 9, 1]

    # Identity matrix
    I = torch.eye(3, device=device).expand(batch_size, 3, 3)

    # Reshape to [N, 3, 3] and add identity
    H = p_.view(batch_size, 3, 3) + I

    return H


def meshgrid(x: torch.Tensor, y: torch.Tensor):
    """
    Create a centered meshgrid from vectors x and y.
    Args:
        x: Tensor of shape [W]
        y: Tensor of shape [H]
    Returns:
        X, Y: Meshgrid tensors of shape [H, W]
    """
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)

    x = x - x.max() / 2
    y = y - y.max() / 2

    X = x.unsqueeze(0).repeat(y.size(0), 1)  # shape [H, W]
    Y = y.unsqueeze(1).repeat(1, x.size(0))  # shape [H, W]

    return X, Y

def warp_hmg(img: torch.Tensor, p: torch.Tensor):
    """
    Warp a batch of images using homography parameters.

    Args:
        img: [N, C, H, W] image batch
        p:   [N, 8, 1] homography parameter batch

    Returns:
        X_warp, Y_warp: warped coordinate grids [N, H, W]
    """
    device = img.device
    batch_size, C, H, W = img.size()

    # Create coordinate grid
    x = torch.arange(W, device=device)
    y = torch.arange(H, device=device)
    X, Y = meshgrid(x, y)  # both [H, W]
    
    # Homogeneous grid: [3, H*W]
    ones = torch.ones(1, X.numel(), device=device)
    xy = torch.cat([
        X.view(1, -1),
        Y.view(1, -1),
        ones
    ], dim=0)  # [3, H*W]

    # Expand to batch: [N, 3, H*W]
    xy = xy.unsqueeze(0).repeat(batch_size, 1, 1)

    # Convert p to homography matrix
    H_mat = param_to_H(p)  # returns [N, 3, 3]

    # Warp: [N, 3, H*W]
    xy_warp = H_mat.bmm(xy)

    # Normalize homogeneous coordinates
    X_warp = xy_warp[:, 0, :] / xy_warp[:, 2, :]
    Y_warp = xy_warp[:, 1, :] / xy_warp[:, 2, :]

    # Reshape and shift grid to original image frame
    X_warp = X_warp.view(batch_size, H, W) + (W - 1) / 2
    Y_warp = Y_warp.view(batch_size, H, W) + (H - 1) / 2

    img_warp, mask = grid_bilinear_sampling(img, X_warp, Y_warp)

    return img_warp, mask


