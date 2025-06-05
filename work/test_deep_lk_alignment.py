import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------------
# 0.  import your code
# ------------------------------------------------------------------------------
import model as dlk

device = "cpu"                   # change to "cuda" if you like
to_tensor = T.ToTensor()

# ------------------------------------------------------------------------------
# 1.  load template + translated frame; normalise like in training
# ------------------------------------------------------------------------------
template = to_tensor(Image.open("template.png")).unsqueeze(0).to(device)
frame    = to_tensor(Image.open("frame.png"   )).unsqueeze(0).to(device)

template = dlk.normalize_img_batch(template)
frame    = dlk.normalize_img_batch(frame)

# ------------------------------------------------------------------------------
# 2.  run LK on raw pixels just to check things work
# ------------------------------------------------------------------------------
dlk_net = dlk.DeepLK(dlk.vgg16Conv()).to(device)

p_est, H_est, itr = dlk_net(frame, template,
                            tol=1e-3, max_itr=300,
                            ret_itr=True, conv_flag=0)

print("estimated Δx, Δy :", p_est[:, [2,5], 0])
print("iterations       :", itr)

warped, _ = dlk.warp_hmg(frame, p_est)          # align frame to template

# visualise
def show_three(tmpl, frm, wrp, title_wrp="After LK alignment"):
    tmpl_np = tmpl.squeeze().cpu().numpy()
    frm_np  = frm.squeeze().cpu().numpy()
    wrp_np  = wrp.squeeze().cpu().numpy()
    overlay = np.stack([tmpl_np, wrp_np, np.zeros_like(tmpl_np)], axis=-1)

    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1); plt.title("Template"); plt.imshow(tmpl_np, cmap='gray'); plt.axis('off')
    plt.subplot(1,3,2); plt.title("Input frame"); plt.imshow(frm_np, cmap='gray'); plt.axis('off')
    plt.subplot(1,3,3); plt.title(title_wrp); plt.imshow(np.clip(overlay,0,1)); plt.axis('off')
    plt.tight_layout(); plt.show()

show_three(template, frame, warped)