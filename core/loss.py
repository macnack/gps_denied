import torch
import torch.nn as nn
from .geometry import param_to_H
import kornia


def corner_loss(
    p: torch.Tensor, p_gt: torch.Tensor, training_sz_pad: float
) -> torch.Tensor:
    """
    Compute corner-based geometric loss between two sets of homography parameters.

    Args:
        p (Tensor):      [N, 8, 1] predicted warp parameters
        p_gt (Tensor):   [N, 8, 1] ground truth warp parameters
        training_sz_pad (float): Side length of the padded image region

    Returns:
        loss (Tensor): Scalar loss (sum over batch)
    """
    device = p.device
    batch_size = p.size(0)

    # Convert to homography matrices
    H_p = param_to_H(p)
    H_gt = param_to_H(p_gt)

    # Define 4 corners of square: [-pad/2, pad/2] range
    corners = (
        torch.tensor(
            [
                [-0.5, 0.5, 0.5, -0.5],  # x
                [-0.5, -0.5, 0.5, 0.5],  # y
                [1.0, 1.0, 1.0, 1.0],  # homogeneous
            ],
            dtype=torch.float32,
            device=device,
        )
        * training_sz_pad
    )  # scale to training size

    # Repeat for batch
    corners = corners.unsqueeze(0).repeat(batch_size, 1, 1)  # [N, 3, 4]

    # Warp corners with predicted and GT homographies
    warped_p = H_p.bmm(corners)  # [N, 3, 4]
    warped_gt = H_gt.bmm(corners)  # [N, 3, 4]

    # Convert from homogeneous to 2D
    warped_p = warped_p[:, :2, :] / warped_p[:, 2:3, :]
    warped_gt = warped_gt[:, :2, :] / warped_gt[:, 2:3, :]

    # Compute squared corner loss
    loss = ((warped_p - warped_gt) ** 2).sum()

    return loss


def four_corner_dist(H_1_2, H_1_2_gt, height, width):
    """
    # L1 distance between 4 corners of source image warped using GT homography
    # and estimated homography transform
    """
    Hinv_gt = torch.inverse(H_1_2_gt)
    Hinv = torch.inverse(H_1_2)
    grid = kornia.utils.create_meshgrid(2, 2, device=Hinv.device)
    warped_grid = kornia.geometry.transform.homography_warper.warp_grid(grid, Hinv)
    warped_grid_gt = kornia.geometry.transform.homography_warper.warp_grid(
        grid, Hinv_gt
    )
    warped_grid = (warped_grid + 1) / 2
    warped_grid_gt = (warped_grid_gt + 1) / 2
    warped_grid[..., 0] *= width
    warped_grid[..., 1] *= height
    warped_grid_gt[..., 0] *= width
    warped_grid_gt[..., 1] *= height
    dist = torch.norm(warped_grid - warped_grid_gt, p=2, dim=-1)
    dist = dist.mean(dim=-1).mean(dim=-1)
    return dist


def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


def matrix_loss(H_pred, H_gt):
    return torch.nn.functional.l1_loss(
        H_pred / H_pred[:, 2:3, 2:3], H_gt / H_gt[:, 2:3, 2:3]
    )


def reprojection_loss(H_pred, H_gt, points):
    warped_pred = kornia.geometry.transform_points(H_pred, points)
    warped_gt = kornia.geometry.transform_points(H_gt, points)
    return torch.nn.functional.l1_loss(warped_pred, warped_gt)


def photometric_loss(I_src, I_tgt, H_pred):
    I_src_warped = kornia.geometry.transform.warp_perspective(
        I_src, H_pred, dsize=I_tgt.shape[-2:]
    )
    return torch.nn.functional.l1_loss(I_src_warped, I_tgt)
