import torch
import torch.nn as nn
from .geometry import param_to_H, warp_perspective_norm, wrap_sRt_norm, param_to_A
import kornia
import theseus as th
from typing import Tuple
from core.error_registry import register


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


@register(
    var_name="H8_1_2",  # Theseus optim-var name
    init_fn=lambda bs, dev: torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], device=dev, dtype=torch.float32
    ).repeat(bs, 1),
    id_vals=torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
    dim=1,  # residual dimensionality
    reshape=(-1, 8),
    get_homography=lambda t: torch.cat([t, t.new_ones(t.shape[0], 1)], dim=-1),
)
def homography_error_fn(optim_vars: Tuple[th.Manifold], aux_vars: Tuple[th.Variable]):
    """
    loss is difference between warped and target image
    """
    H8_1_2 = optim_vars[0].tensor.reshape(-1, 8)

    # Force the last element H[2,2] to be 1.
    H_1_2 = torch.cat(
        [H8_1_2, H8_1_2.new_ones(H8_1_2.shape[0], 1)], dim=-1
    )  # type: ignore

    img1: th.Variable = aux_vars[0]
    img2: th.Variable = aux_vars[-1]

    img1_dst = warp_perspective_norm(H_1_2.reshape(-1, 3, 3), img1.tensor)

    loss = torch.nn.functional.mse_loss(img1_dst, img2.tensor, reduction="none")

    one_with_zero_boarder = torch.zeros_like(img1.tensor)
    one_with_zero_boarder[:, :, 1:-1, 1:-1] = 1.0

    ones = warp_perspective_norm(
        H_1_2.data.reshape(-1, 3, 3), one_with_zero_boarder
    )

    mask = ones > 0.9
    loss = loss.view(loss.shape[0], -1)
    mask = mask.view(loss.shape[0], -1)
    loss = (loss * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
    return loss


@register(
    var_name="A4_1_2",
    init_fn=lambda bs, dev: torch.tensor(
        [[1.0, 0.00, 0.0, 0.0]], device=dev, dtype=torch.float32
    ).repeat(bs, 1),
    id_vals=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    dim=1,
    reshape=(-1, 4),
    get_homography=lambda t: kornia.geometry.convert_affinematrix_to_homography(
        param_to_A(t)
    ),
)
def sRt_error_fn(optim_vars: Tuple[th.Manifold], aux_vars: Tuple[th.Variable]):

    A4_1 = optim_vars[0].tensor.reshape(-1, 4)

    img1: th.Variable = aux_vars[0]
    img2: th.Variable = aux_vars[-1]
    img1_dst = wrap_sRt_norm(A4_1, img1.tensor)

    loss = torch.nn.functional.huber_loss(img1_dst, img2.tensor, reduction="none")

    one_with_zero_boarder = torch.zeros_like(img1.tensor)
    one_with_zero_boarder[:, :, 1:-1, 1:-1] = 1.0

    ones = wrap_sRt_norm(A4_1, one_with_zero_boarder)

    mask = torch.sigmoid( (ones - 0.90) * 10 )
    
    loss = loss.view(loss.shape[0], -1)
    mask = mask.view(loss.shape[0], -1)

    loss = (loss * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1)
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)

    return loss * 100.0


@register(
    var_name="A3_1_2",
    init_fn=lambda bs, dev: torch.tensor(
        [[0.00, 0.0, 0.0]], device=dev, dtype=torch.float32
    ).repeat(bs, 1),
    id_vals=torch.tensor([[0.0, 0.0, 0.0]]),
    dim=1,
    reshape=(-1, 3),
    get_homography=lambda t: kornia.geometry.convert_affinematrix_to_homography(
        param_to_A(torch.cat([torch.ones((t.shape[0], 1), device=t.device), t], dim=1))
    ),
)
def Rt_error_fn(optim_vars: Tuple[th.Manifold], aux_vars: Tuple[th.Variable]):

    A3 = optim_vars[0].tensor.reshape(-1, 3)
    scale = torch.ones((A3.shape[0], 1), device=A3.device)
    A4_1 = torch.cat([scale, A3], dim=1)

    img1: th.Variable = aux_vars[0]
    img2: th.Variable = aux_vars[-1]
    img1_dst = wrap_sRt_norm(A4_1, img1.tensor)

    loss = torch.nn.functional.huber_loss(img1_dst, img2.tensor, reduction="none")

    one_with_zero_boarder = torch.zeros_like(img1.tensor)
    one_with_zero_boarder[:, :, 1:-1, 1:-1] = 1.0

    ones = wrap_sRt_norm(A4_1, one_with_zero_boarder)

    mask = torch.sigmoid( (ones - 0.90) * 10 )
    
    loss = loss.view(loss.shape[0], -1)
    mask = mask.view(loss.shape[0], -1)

    loss = (loss * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1)
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)

    return loss * 100.0


class CornerLoss(nn.Module):
    """
    Compute corner‐based geometric loss between two sets of homography parameters.

    Given two homographies (predicted and ground‐truth) for a square of side
    length `training_sz_pad`, this loss warps the four corners of the square
    through each homography and returns the sum of squared distances between
    the warped corners.

    Args:
        training_sz_pad (float):
            Side length of the padded image region (so that corners lie at
            ±training_sz_pad/2 in x,y).
    """

    def __init__(self, training_sz_pad: float):
        super().__init__()
        self.training_sz_pad = float(training_sz_pad)

        # Define 4 corners in normalized coordinates (±0.5 in x,y, homogeneous=1)
        # We register this as a buffer so it automatically lives on the correct device.
        base_corners = torch.tensor(
            [
                [-0.5, 0.5, 0.5, -0.5],  # x
                [-0.5, -0.5, 0.5, 0.5],  # y
                [1.0, 1.0, 1.0, 1.0],  # homogeneous
            ],
            dtype=torch.float32,
        )  # shape: [3, 4]
        self.register_buffer("base_corners", base_corners)

    def forward(self, p: torch.Tensor, p_gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p (Tensor):      [N, 8, 1] predicted warp parameters
            p_gt (Tensor):   [N, 8, 1] ground-truth warp parameters

        Returns:
            loss (Tensor): Scalar loss (sum over batch & all 4 corners).
        """
        # shapes:
        #   p, p_gt = [N, 8, 1]
        #   base_corners = [3, 4]

        device = p.device
        batch_size = p.size(0)

        # Convert 8-vector parameters to 3×3 homography matrices:
        #   H  →  shape [N, 3, 3]
        H_p = param_to_H(p)
        H_gt = param_to_H(p_gt)

        # Build a [N, 3, 4] tensor of “true” corners (scaled to ±pad/2 in x,y).
        # We clone base_corners so we don’t modify the buffer in place.
        corners = self.base_corners.clone().to(device)  # [3, 4]
        # Scale only x & y by training_sz_pad; leave homogeneous row = 1
        corners[:2, :] = corners[:2, :] * self.training_sz_pad
        # Now corners = [[–pad/2, +pad/2, +pad/2, –pad/2],
        #               [–pad/2, –pad/2, +pad/2, +pad/2],
        #               [   1.0,    1.0,    1.0,    1.0 ]]
        # shape = [3, 4]
        corners = corners.unsqueeze(0).repeat(batch_size, 1, 1)  # [N, 3, 4]

        # Warp each batch’s corners through H_p and H_gt
        # warped_p, warped_gt both have shape [N, 3, 4]
        warped_p = H_p.bmm(corners)  # each [3×3] × [3×4] → [3×4]
        warped_gt = H_gt.bmm(corners)

        # Convert from homogeneous to 2D:  (x’/w’, y’/w’) for each corner
        # After division, warped_p_2d, warped_gt_2d have shape [N, 2, 4].
        warped_p_2d = warped_p[:, :2, :] / warped_p[:, 2:3, :]
        warped_gt_2d = warped_gt[:, :2, :] / warped_gt[:, 2:3, :]

        # Sum of squared distances across all corners and batch
        # → scalar tensor
        loss = ((warped_p_2d - warped_gt_2d) ** 2).sum()
        # # Normalize by batch size???
        # loss = loss / float(batch_size)
        return loss


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
