from core.error_registry import register
import theseus as th
from typing import Tuple
import torch
from .geometry import warp_perspective_norm, wrap_sRt_norm, param_to_A
import kornia


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

    ones = warp_perspective_norm(H_1_2.data.reshape(-1, 3, 3), one_with_zero_boarder)

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

    mask = torch.sigmoid((ones - 0.90) * 10)

    loss = loss.view(loss.shape[0], -1)
    mask = mask.view(loss.shape[0], -1)

    loss = (loss * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(
        min=1
    )
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

    mask = torch.sigmoid((ones - 0.90) * 10)

    loss = loss.view(loss.shape[0], -1)
    mask = mask.view(loss.shape[0], -1)

    loss = (loss * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(
        min=1
    )
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)

    return loss * 100.0
