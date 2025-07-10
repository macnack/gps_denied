from PIL import Image
import kornia
import torch
import numpy as np
import theseus as th
from theseus.core.cost_function import ErrFnType
from core.loss import homography_error_fn, four_corner_dist
from typing import Any, Dict, List, Optional, Tuple, Type, cast
import os
from core.viz import write_gif_batch
from core.error_registry import get as get_spec

img_dir = "tests/img_t.png"


if __name__ == "__main__":
    img = Image.open(img_dir).convert("RGB")
    width, height = 100, 100
    img = img.resize((width, height))
    spec = get_spec("homography_error_fn")     
    H_rot = [[0, -1, img.size[0]], [1, 0, 0], [0, 0, 1]]

    center = torch.tensor([[width / 2.0, height / 2.0]])
    angle_deg = 5.0  # rotation angle in degrees

    # Get 2x3 affine rotation matrix
    angle_rad = torch.tensor([angle_deg])
    scale = torch.ones((1, 2))
    R = kornia.geometry.transform.get_rotation_matrix2d(center, angle_rad, scale)  # (B, 2, 3)
    H_rot = kornia.geometry.convert_affinematrix_to_homography(R)  # (B, 3, 3)

    np_img = np.array(img)
    
    tensor_img = kornia.utils.image_to_tensor(np_img, keepdim=False).float() / 255.0
    tensor_img = kornia.filters.gaussian_blur2d(
        tensor_img, (1, 1), (5, 5))
    wraped_img_tensor = kornia.geometry.transform.warp_perspective(
        tensor_img, 
        H_rot, 
        dsize=(img.size[1], img.size[0])
    )
    wraped_img = wraped_img_tensor.squeeze(0)  # Remove batch dimension
    wraped_img = kornia.utils.tensor_to_image(wraped_img)
    wraped_img = Image.fromarray((wraped_img * 255).astype(np.uint8))
    # img.show()
    # wraped_img.show()
    H8_init = spec.init_fn(1, torch.device("cpu"))
    objective = th.Objective()
    feats = torch.zeros_like(tensor_img)
    H8_1_2 = th.Vector(tensor=H8_init, name=spec.var_name)
    feat1 = th.Variable(tensor=feats, name="feat1")
    feat2 = th.Variable(tensor=feats, name="feat2")
    
    autograd_mode: str = "vmap"
    homography_cf = th.AutoDiffCostFunction(
        optim_vars=[H8_1_2],
        err_fn=cast(ErrFnType, homography_error_fn),
        dim=spec.dim,
        aux_vars=[feat1, feat2],
        autograd_mode=autograd_mode,
    )
    objective.add(homography_cf)
    
    reg_w_value = 1e-2
    reg_w = th.ScaleCostWeight(np.sqrt(reg_w_value))
    reg_w.to(dtype=H8_init.dtype)
    vals = spec.id_vals
    H8_1_2_id = th.Vector(tensor=vals, name="identity")
    reg_cf = th.Difference(
        H8_1_2, target=H8_1_2_id, cost_weight=reg_w, name="reg_homography"
    )
    objective.add(reg_cf)
    linear_solver_info = None
    max_iterations = 300
    step_size = 1e-3
    verbose = True
    device = "cpu"
    if linear_solver_info is not None:
        linear_solver_cls, linearization_cls = linear_solver_info
    else:
        linear_solver_cls, linearization_cls = None, None
    inner_optim = th.LevenbergMarquardt(
        objective,
        linear_solver_cls=linear_solver_cls,
        linearization_cls=linearization_cls,
        max_iterations=max_iterations,
        step_size=step_size,
    )
    
    inputs: Dict[str, torch.Tensor] = {
                spec.var_name : H8_init,
                "feat1": tensor_img,
                "feat2": wraped_img_tensor,
            }
    theseus_layer = th.TheseusLayer(inner_optim).to(device)
    _, info = theseus_layer.forward(
                    inputs,
                    optimizer_kwargs={
                        "verbose": verbose,
                        "track_err_history": True,
                        "track_state_history": True,
                        "backward_mode": "implicit",
                    },
                )
    
    H8_1_2_tensor = theseus_layer.objective.get_optim_var(
                spec.var_name
            ).tensor.reshape(spec.reshape)
    H_1_2 = spec.get_homography(H8_1_2_tensor)
    fc_dist = four_corner_dist(
                H_1_2.reshape(-1, 3, 3), H_rot.reshape(-1, 3, 3), img.size[1], img.size[0]
            )
    optimizer_info: th.NonlinearOptimizerInfo = cast(
                    th.NonlinearOptimizerInfo, info
                )
    H_hist = optimizer_info.state_history
    err_hist = optimizer_info.err_history
    log_dir = os.path.join(os.getcwd(), "viz")

    print(f"Final 4-corner distance: {fc_dist.item()}")
    print(f"Final Homography: {H_1_2}")
    print(f"Final Homography (H8): {H_rot}")
    write_gif_batch(log_dir, feat1, feat2, H_hist[spec.var_name], H_rot, err_hist, func=spec.get_homography)
    print(f"GIF saved to {log_dir}/animation.gif")

    print(err_hist[0][-1].item())
    print(err_hist[0].mean().item())
    print(err_hist[0].min().item())
    print(err_hist[0].max().item())
