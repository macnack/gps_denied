import torch
from .sampling import grid_bilinear_sampling
from theseus.third_party.utils import grid_sample
import kornia


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
    xy = torch.cat([X.view(1, -1), Y.view(1, -1), ones], dim=0)  # [3, H*W]

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


def warp_perspective_norm(H, img):
    height, width = img.shape[-2:]
    grid = kornia.utils.create_meshgrid(
        height, width, normalized_coordinates=True, device=H.device
    )
    Hinv = torch.inverse(H)
    warped_grid = kornia.geometry.transform.homography_warper.warp_grid(
        grid, Hinv)
    # Using custom implementation, above will throw error with outer loop optim.
    img2 = grid_sample(img, warped_grid)
    return img2


def param_to_A(p: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 4-parameter vectors to (2, 3) affine matrices.

    Args:
        p: Tensor of shape [N, 4] where
           p[:, 0] = scale (shared for x & y)
           p[:, 1] = rotation term
           p[:, 2] = translation x
           p[:, 3] = translation y
    Returns:
        Tensor [N, 2, 3] with rows
        [[ s,  r,  tx ],
         [ 0,  s,  ty ]]
    """
    s, r, tx, ty = p.unbind(dim=1)

    center_x = 0.0  # FIXME: should be dynamic W/2
    center_y = 0.0  # FIXME: should be dynamic H/2
    r_scaled = r * (torch.pi / 180.0)  # Convert degrees to radians
    cos_r = torch.cos(r_scaled)
    sin_r = torch.sin(r_scaled)

    sf = torch.ones_like(s)

    a = sf * cos_r
    b = -sf * sin_r
    c = sf * sin_r
    d = sf * cos_r

    # The translation part is modified to perform rotation around the center
    # t_final = t + C - sR*C
    tx_final = tx + center_x - (a * center_x + b * center_y)
    ty_final = ty + center_y - (c * center_x + d * center_y)

    return torch.stack(
        (
            a, b, tx_final,
            c, d, ty_final
        ),
        dim=-1
    ).view(-1, 2, 3)


def wrap_sRt_norm(A, img):
    """
    Wrap an image using a 3x4 matrix A (sRt) with normalized coordinates.
    Args:
        A: [N, 4] matrix [scale, rotation, dx, dy]
        img: [N, C, H, W] image batch
    Returns:
        img2: warped image
    """

    H = kornia.geometry.convert_affinematrix_to_homography(param_to_A(A))

    return warp_perspective_norm(H, img)
