import torch
import torch.nn.functional as F

def grid_bilinear_sampling(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    """
    Bilinear sampling of A at (x, y) locations using grid_sample.

    Args:
        A: [N, C, H, W] input feature map
        x: [N, H, W] x-coordinates to sample
        y: [N, H, W] y-coordinates to sample

    Returns:
        Q: [N, C, H, W] sampled output
        in_view_mask: [N, H, W] binary mask of valid samples
    """
    batch_size, C, H, W = A.size()

    # Normalize coordinates to [-1, 1]
    x_norm = x / ((W - 1) / 2) - 1
    y_norm = y / ((H - 1) / 2) - 1

    # Stack into grid: shape [N, H, W, 2]
    grid = torch.stack((x_norm, y_norm), dim=-1)

    # Perform bilinear sampling
    Q = F.grid_sample(A, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # Compute in-view mask (inside [-1+ε, 1−ε])
    eps_w = 2 / W
    eps_h = 2 / H
    in_view_mask = (
        (x_norm > -1 + eps_w) & (x_norm < 1 - eps_w) &
        (y_norm > -1 + eps_h) & (y_norm < 1 - eps_h)
    ).to(dtype=A.dtype)

    return Q, in_view_mask


def normalize_img_batch(img: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of images to zero mean and unit variance per channel.

    Args:
        img: Tensor of shape [N, C, H, W]

    Returns:
        Normalized image batch of same shape
    """
    N, C, H, W = img.shape

    # Flatten spatial dimensions
    img_vec = img.view(N, C, -1)

    # Compute per-image per-channel mean and std
    mean = img_vec.mean(dim=2, keepdim=True)
    std = img_vec.std(dim=2, keepdim=True)

    # Avoid division by zero
    std = std + 1e-8

    # Normalize
    img_norm = (img_vec - mean) / std

    return img_norm.view(N, C, H, W)
