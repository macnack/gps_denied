from PIL import Image
import kornia
import torch
import numpy as np
import theseus as th
from theseus.core.cost_function import ErrFnType
from core.loss import homography_error_fn, four_corner_dist, sRt_error_fn, param_to_A, wrap_sRt_norm
from core.geometry import warp_perspective_norm
from typing import Any, Dict, List, Optional, Tuple, Type, cast
import os
from core.viz import write_gif_batch, visualize_corner_loss
from core.error_registry import get as get_spec

img_dir = "tests/ukraine_post.jpg"


if __name__ == "__main__":
    img = Image.open(img_dir).convert("RGB")
    spec = get_spec("sRt_error_fn")

    def crop(img, x_center, y_center, width, height):
        left = x_center - width // 2
        upper = y_center - height // 2
        right = x_center + width // 2
        lower = y_center + height // 2
        return img.crop((left, upper, right, lower))

    img1 = crop(img, img.size[0] // 2, img.size[1] - 1000, 200, 200)

    width, height = 100, 100
    img = img.resize((width, height))

    # [scale, rotation, translation_x, translation_y]
    A_true = torch.tensor([[1.0, 0.4, -0.25, 0.30]], dtype=torch.float32)
    H_rot = kornia.geometry.convert_affinematrix_to_homography(
        param_to_A(A_true))

    np_img = np.array(img).astype(np.float32)

    tensor_img = kornia.utils.image_to_tensor(
        np_img, keepdim=False).float() / 255.0

    # tensor_img = kornia.filters.box_blur(tensor_img, (3, 3))
    gaussian_blur = kornia.filters.GaussianBlur2d(
        kernel_size=(5, 5), sigma=(3.0, 3.0))
    tensor_img = gaussian_blur(tensor_img)
    # tensor_img = kornia.filters.sobel(tensor_img, normalized=True)

    wraped_img_tensor = warp_perspective_norm(
        H_rot,
        tensor_img)

    wraped_img = wraped_img_tensor.squeeze(0)  # Remove batch dimension
    wraped_img = kornia.utils.tensor_to_image(wraped_img)
    wraped_img = Image.fromarray((wraped_img * 255).astype(np.uint8))
    # wraped_img.show()

    # H8_init = torch.eye(3).reshape(1, 9)[:, :-1].repeat(1, 1)
    A4_init = spec.id_vals
    objective = th.Objective()
    feats = torch.zeros_like(tensor_img)
    A4_1_2 = th.Vector(tensor=A4_init, name=spec.var_name)
    feat1 = th.Variable(tensor=feats, name="feat1")
    feat2 = th.Variable(tensor=feats, name="feat2")

    autograd_mode: str = "vmap"
    homography_cf = th.AutoDiffCostFunction(
        optim_vars=[A4_1_2],
        err_fn=cast(ErrFnType, sRt_error_fn),
        dim=spec.dim,
        aux_vars=[feat1, feat2],
        autograd_mode=autograd_mode,
    )
    objective.add(homography_cf)

    # reg_w_value = 1e-2
    # reg_w = th.ScaleCostWeight(np.sqrt(reg_w_value))
    # reg_w.to(dtype=A4_init.dtype)
    # vals = spec.id_vals
    # H8_1_2_id = th.Vector(tensor=vals, name="identity")
    # reg_cf = th.Difference(
    #     A4_1_2, target=H8_1_2_id, cost_weight=reg_w, name="reg_homography"
    # )
    # objective.add(reg_cf)
    linear_solver_info = None
    max_iterations = 25
    step_size = 6e-2
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
        abs_err_tolerance=1e-8,
        rel_err_tolerance=1e-10
    )
    # TODO test if it is better to stop earlier for better convergence of outer loop
    # e.g higher rel_err_tolerance or lower max_iterations

    inputs: Dict[str, torch.Tensor] = {
        spec.var_name : A4_init,
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

    A4_1_2_tensor = theseus_layer.objective.get_optim_var(
        spec.var_name
    ).tensor.reshape(spec.reshape)

    H_1_2 = spec.get_homography(A4_1_2_tensor)

    fc_dist = four_corner_dist(
        H_1_2.reshape(-1, 3, 3),
        H_rot.reshape(-1, 3, 3),
        img.size[1], img.size[0]
    )

    optimizer_info: th.NonlinearOptimizerInfo = cast(
        th.NonlinearOptimizerInfo, info
    )
    H_hist_pre = optimizer_info.state_history
    print(f"State history length: {H_hist_pre['A4_1_2'].shape}")

    err_hist = optimizer_info.err_history
    log_dir = os.path.join(os.getcwd(), "viz")

    print(f"Final 4-corner distance: {fc_dist.item()}")

    print(f"Final Homography Est: {H_1_2}")
    print(f"Final Homography GT (H8): {H_rot}")

    print(f"Final EST: {A4_1_2_tensor}")
    print(f"Orginal Homography GT: {A_true}")
    H_1_2 = H_1_2.unsqueeze(0)
    
    H_rot = H_rot.unsqueeze(0)
    
    print("shape of H_1_2:", H_rot.shape)   
    visualize_corner_loss(H_1_2, H_rot, training_sz_pad=10)

    write_gif_batch(log_dir, feat1, feat2, H_hist_pre[spec.var_name], H_rot, err_hist, func=spec.get_homography)
    print(f"GIF saved to {log_dir}/animation.gif")

    print(err_hist[0][-1].item())
    print(err_hist[0].mean().item())
    print(err_hist[0].min().item())
    print(err_hist[0].max().item())
    

