#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------
# helpers – you already have something like these in your repo
# -------------------------------------------------------------
def build_centered_h(angle_deg, tx_px, ty_px, h, w):
    """
    return a 3×3 homography in *centre* coords
    """
    theta = np.deg2rad(angle_deg)
    H_pixel = np.array([[ np.cos(theta), -np.sin(theta), tx_px],
                        [ np.sin(theta),  np.cos(theta), ty_px],
                        [      0.,             0.,       1. ]], dtype=np.float32)

    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    T      = np.array([[1, 0, -cx],
                       [0, 1, -cy],
                       [0, 0,   1]], dtype=np.float32)
    T_inv  = np.array([[1, 0,  cx],
                       [0, 1,  cy],
                       [0, 0,   1]], dtype=np.float32)

    return T @ H_pixel @ T_inv     # H_center


def homography_to_grid(H, h, w, device):
    """
    H : 3×3 (numpy or tensor on cpu)
    returns grid tensor [1,H,W,2] in [-1,1] suitable for grid_sample
    """
    if isinstance(H, np.ndarray):
        H = torch.from_numpy(H)

    H = H.to(device)

    # pixel grid
    ys, xs = torch.meshgrid(torch.arange(h, device=device),
                            torch.arange(w, device=device),
                            indexing='ij')
    xs = xs.flatten().float()                # (H*W,)
    ys = ys.flatten().float()

    # to centre coords
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    x_c = xs - cx
    y_c = ys - cy
    ones = torch.ones_like(x_c)

    xy1 = torch.stack([x_c, y_c, ones], dim=0)      # 3 × (H*W)
    xyz = H @ xy1                                   # 3 × (H*W)

    xs_w = xyz[0] / xyz[2] + cx                     # back to pixel coords
    ys_w = xyz[1] / xyz[2] + cy

    # normalise to [-1,1] (align_corners=True)
    xs_n = 2.0 * xs_w / (w - 1) - 1.0
    ys_n = 2.0 * ys_w / (h - 1) - 1.0

    grid = torch.stack([xs_n, ys_n], dim=1)         # (H*W, 2)
    grid = grid.view(1, h, w, 2)                    # 1×H×W×2
    return grid


# -------------------------------------------------------------
# 1. load image
# -------------------------------------------------------------
device   = "cpu"
to_tensor = T.ToTensor()
template = to_tensor(Image.open("template.png")).unsqueeze(0).to(device)  # 1×1×H×W

# -------------------------------------------------------------
# 2. build known homography   ( +20°,  +10  px right,  −4 px up )
# -------------------------------------------------------------
H, W = template.shape[-2:]
H_center = build_centered_h(angle_deg=20, tx_px=10, ty_px=-4, h=H, w=W)

# -------------------------------------------------------------
# 3. convert to grid and warp with grid_sample
# -------------------------------------------------------------
grid = homography_to_grid(H_center, H, W, device)
warped = F.grid_sample(template, grid,
                       mode='bilinear',
                       padding_mode='zeros',
                       align_corners=True)          # 1×1×H×W

# -------------------------------------------------------------
# 4. overlay for visual sanity-check
# -------------------------------------------------------------
tmpl_np  = template.squeeze().cpu().numpy()
warp_np  = warped.squeeze().cpu().numpy()
overlay  = np.stack([tmpl_np, warp_np, np.zeros_like(tmpl_np)], axis=-1)

plt.figure(figsize=(9,3))
plt.subplot(1,3,1); plt.title("Template");  plt.imshow(tmpl_np, cmap='gray');  plt.axis('off')
plt.subplot(1,3,2); plt.title("Warped");    plt.imshow(warp_np, cmap='gray');  plt.axis('off')
plt.subplot(1,3,3); plt.title("Overlay");   plt.imshow(np.clip(overlay,0,1));  plt.axis('off')
plt.tight_layout(); plt.show()
