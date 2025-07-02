import torch
import torch.nn as nn
from .geometry import param_to_H

def corner_loss(p: torch.Tensor, p_gt: torch.Tensor, training_sz_pad: float) -> torch.Tensor:
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
    corners = torch.tensor([
        [-0.5,  0.5,  0.5, -0.5],  # x
        [-0.5, -0.5,  0.5,  0.5],  # y
        [ 1.0,  1.0,  1.0,  1.0]   # homogeneous
    ], dtype=torch.float32, device=device) * training_sz_pad  # scale to training size

    # Repeat for batch
    corners = corners.unsqueeze(0).repeat(batch_size, 1, 1)  # [N, 3, 4]

    # Warp corners with predicted and GT homographies
    warped_p = H_p.bmm(corners)    # [N, 3, 4]
    warped_gt = H_gt.bmm(corners)  # [N, 3, 4]

    # Convert from homogeneous to 2D
    warped_p = warped_p[:, :2, :] / warped_p[:, 2:3, :]
    warped_gt = warped_gt[:, :2, :] / warped_gt[:, 2:3, :]

    # Compute squared corner loss
    loss = ((warped_p - warped_gt) ** 2).sum()

    return loss

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
        base_corners = torch.tensor([
            [-0.5,  0.5,  0.5, -0.5],   # x
            [-0.5, -0.5,  0.5,  0.5],   # y
            [ 1.0,  1.0,  1.0,  1.0]    # homogeneous
        ], dtype=torch.float32)  # shape: [3, 4]
        self.register_buffer('base_corners', base_corners)

    def forward(self,
                p: torch.Tensor,
                p_gt: torch.Tensor) -> torch.Tensor:
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
        H_p  = param_to_H(p)
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
        warped_p  = H_p .bmm(corners)   # each [3×3] × [3×4] → [3×4]
        warped_gt = H_gt.bmm(corners)

        # Convert from homogeneous to 2D:  (x’/w’, y’/w’) for each corner
        # After division, warped_p_2d, warped_gt_2d have shape [N, 2, 4].
        warped_p_2d  = warped_p [:, :2, :] / warped_p [:, 2:3, :]
        warped_gt_2d = warped_gt[:, :2, :] / warped_gt[:, 2:3, :]

        # Sum of squared distances across all corners and batch
        # → scalar tensor
        loss = ((warped_p_2d - warped_gt_2d) ** 2).sum()
        # # Normalize by batch size???
        # loss = loss / float(batch_size)
        return loss