import numpy as np
import cv2
import logging
from torch import abs, cat
from .loss import four_corner_dist
from .geometry import warp_perspective_norm
import os
import shutil
from torch import Tensor
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SZ = 0.5
FONT_PT = (5, 25)

logger = logging.getLogger(__name__)


def put_text(img, text, top=True):
    if top:
        pt = FONT_PT
    else:
        pt = FONT_PT[0], int(img.shape[0] * 1.08 - FONT_PT[1])
    cv2.putText(img, text, pt, FONT, FONT_SZ, (255, 255, 255), 2, lineType=16)
    cv2.putText(img, text, pt, FONT, FONT_SZ, (0, 0, 0), 1, lineType=16)
    return img


def torch2cv2(img):
    out = (img.permute(1, 2, 0) * 255.0).data.cpu().numpy().astype(np.uint8)[:, :, ::-1]
    out = np.ascontiguousarray(out)
    return out


def viz_warp(path, img1, img2, img1_w, iteration, err=-1.0, fc_err=-1.0):
    img_diff = tensor_to_cv2(abs(img1_w - img2))
    img1 = tensor_to_cv2(img1)
    img2 = tensor_to_cv2(img2)
    img1_w = tensor_to_cv2(img1_w)
    factor = 2
    new_sz = int(factor * img1.shape[1]), int(factor * img1.shape[0])
    img1 = cv2.resize(img1, new_sz, interpolation=cv2.INTER_NEAREST)
    img1 = cv2.resize(img1, new_sz)
    img2 = cv2.resize(img2, new_sz, interpolation=cv2.INTER_NEAREST)
    img1_w = cv2.resize(img1_w, new_sz, interpolation=cv2.INTER_NEAREST)
    img_diff = cv2.resize(img_diff, new_sz, interpolation=cv2.INTER_NEAREST)
    img1 = put_text(img1, "image I")
    img2 = put_text(img2, "image I'")
    img1_w = put_text(img1_w, "I warped to I'")
    img_diff = put_text(img_diff, "L2 diff")
    out = np.concatenate([img1, img2, img1_w, img_diff], axis=1)
    out = put_text(
        out,
        "iter: %05d, loss: %.8f, fc_err: %.3f px" % (iteration, err, fc_err),
        top=False,
    )
    cv2.imwrite(path, out)


# write gif showing source image being warped onto target through optimisation
def write_gif_batch(log_dir, img1, img2, H_hist, Hgt_1_2, err_hist, name="animation"):
    anim_dir = f"{log_dir}/{name}"
    os.makedirs(anim_dir, exist_ok=True)
    subsample_anim = 1
    H8_1_2_hist = H_hist["H8_1_2"]
    num_iters = (~err_hist[0].isinf()).sum().item()
    for it in range(num_iters):
        if it % subsample_anim != 0:
            continue
        # Visualize only first element in batch.
        H8_1_2 = H8_1_2_hist[..., it]
        H_1_2 = cat([H8_1_2, H8_1_2.new_ones(H8_1_2.shape[0], 1)], dim=-1).to(
            Hgt_1_2.device
        )
        H_1_2_mat = H_1_2[0].reshape(1, 3, 3)
        Hgt_1_2_mat = Hgt_1_2[0].reshape(1, 3, 3)
        imgH, imgW = img1.shape[-2], img1.shape[-1]
        fc_err = four_corner_dist(H_1_2_mat, Hgt_1_2_mat, imgH, imgW)
        err = float(err_hist[0][it])
        img1 = img1[0][None, ...]
        img2 = img2[0][None, ...]
        img1_dsts = warp_perspective_norm(H_1_2_mat, img1)
        path = os.path.join(anim_dir, f"{it:05d}.png")
        viz_warp(path, img1[0], img2[0], img1_dsts[0], it, err=err, fc_err=fc_err)
    anim_path = os.path.join(log_dir, f"{name}.gif")
    cmd = f"convert -delay 10 -loop 0 {anim_dir}/*.png {anim_path}"
    logger.info("Generating gif here: %s" % anim_path)
    os.system(cmd)
    shutil.rmtree(anim_dir)
    return


def tensor_to_cv2(img: Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor (CHW, RGB, 0-1 floats) to an OpenCV image
    (HWC, BGR, uint8).

    Parameters
    ----------
    img : torch.Tensor
        Tensor with shape [3, H, W] or [1, 3, H, W].  Values are assumed to be
        floats in [0, 1].  Tensor can live on GPU.

    Returns
    -------
    np.ndarray
        OpenCV-compatible image, shape [H, W, 3], dtype uint8, BGR channel order.
    """
    # (1) accept 4-D tensors and squeeze the batch dimension
    if img.dim() == 4:
        img = img.squeeze(0)

    if img.dim() != 3 or img.size(0) != 3:
        raise ValueError("Expecting a tensor of shape [3, H, W] or [1, 3, H, W]")

    # (2) move to CPU, clamp to [0, 1] and convert to uint8
    img_np = (
        img.detach()          # break the graph if it came from a model
           .clamp(0, 1)       # just in case training artefacts pushed it out of range
           .mul_(255)         # scale to [0, 255]
           .byte()            # uint8
           .cpu()             # GPU -> host
           .numpy()           # Torch -> NumPy
    )

    # (3) channels-first (C, H, W)  ->  channels-last (H, W, C)
    img_np = np.transpose(img_np, (1, 2, 0))

    # (4) RGB  ->  BGR (what OpenCV expects)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_np