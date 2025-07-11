import numpy as np
import cv2
import logging
from torch import abs, cat
from .loss import four_corner_dist
from .geometry import warp_perspective_norm
import os
import shutil
from torch import Tensor
import torch
import matplotlib.pyplot as plt

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


def get_homography(tensor):
    raise ValueError(
        "You must pass a 'func' argument that translates the homography tensor (e.g., func = your_tensor_converter_function)"
    )


# write gif showing source image being warped onto target through optimisation
def write_gif_batch(log_dir, img1, img2, H_hist, Hgt_1_2, err_hist, name="animation", func=get_homography):
    anim_dir = f"{log_dir}/{name}"
    os.makedirs(anim_dir, exist_ok=True)
    subsample_anim = 1
    H8_1_2_hist = H_hist
    num_iters = (~err_hist[0].isinf()).sum().item()
    for it in range(num_iters):
        if it % subsample_anim != 0:
            continue
        # Visualize only first element in batch.
        H8_1_2 = H8_1_2_hist[..., it]
        H_1_2 = func(H8_1_2).to(Hgt_1_2.device)
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


def visualize_corner_loss(
    H_p: torch.Tensor,
    H_gt: torch.Tensor,
    training_sz_pad: float,
    image: np.ndarray = None,
):
    """
    Visualize corner-based loss between predicted and GT homographies.

    Args:
        p (Tensor):            [8] or [1,8,1] predicted warp parameters
        p_gt (Tensor):         same shape for ground truth
        training_sz_pad (float): side length of your (padded) image
        param_to_H (callable): function mapping p->[1,3,3] homography
        image (ndarray, optional): H×W×3 array to overlay corners on

    Produces two plots:
      • scatter overlay of GT vs. predicted corners (with connecting lines)
      • bar chart of squared‐distance error per corner
    """
    # -- prepare h/c deco
    H_p = H_p.squeeze(0)
    H_gt = H_gt.squeeze(0)
    # make 4 corners in homogeneous coords ([3,4])
    corners = (
        torch.tensor(
            [
                [-0.5, 0.5, 0.5, -0.5],  # x coords
                [-0.5, -0.5, 0.5, 0.5],  # y coords
                [1.0, 1.0, 1.0, 1.0],
            ],
            device=H_p.device,
        )
        * training_sz_pad
    )

    # warp
    warped_p = H_p.bmm(corners.unsqueeze(0))[0]
    warped_gt = H_gt.bmm(corners.unsqueeze(0))[0]

    # to 2D
    wp = (warped_p[:2] / warped_p[2:]).cpu().numpy().T  # (4,2)
    wgt = (warped_gt[:2] / warped_gt[2:]).cpu().numpy().T

    # per‐corner squared error
    errors = np.sum((wp - wgt) ** 2, axis=1)

    # -- Plot 1: overlay
    plt.figure()
    if image is not None:
        plt.imshow(image)
    plt.scatter(wgt[:, 0], wgt[:, 1], label="GT corners")
    plt.scatter(wp[:, 0], wp[:, 1], label="Predicted")
    for i in range(4):
        plt.plot([wgt[i, 0], wp[i, 0]], [wgt[i, 1], wp[i, 1]])
    plt.legend()
    plt.title("Corner positions: GT vs. Predicted")
    plt.axis("equal")

    # -- Plot 2: bar chart of errors
    plt.figure()
    plt.bar(range(4), errors)
    plt.xlabel("Corner index")
    plt.ylabel("Squared error")
    plt.title("Per‐corner squared error")

    plt.show()
