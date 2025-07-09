# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
import pathlib
import shutil
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import cv2
import hydra
import kornia
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import theseus as th
from theseus.core.cost_function import ErrFnType
from theseus.third_party.easyaug import GeoAugParam, RandomGeoAug, RandomPhotoAug
from theseus.third_party.utils import grid_sample

from datasets import HomographyDataset, prepare_data, ImageDataset
from models import SimpleCNN, DeepCNN, vgg16Conv
from core import (
    homography_error_fn,
    four_corner_dist,
    write_gif_batch,
    compute_grad_norm,
    normalize_img_batch,
)
import neptune
from torch.utils.data import random_split

if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=UserWarning)

# Logger
logger = logging.getLogger(__name__)


def run(
    log,
    model_cfg,
    batch_size: int = 2,
    num_epochs: int = 20,
    outer_lr: float = 1e-4,
    device_param: int = 0,
    max_iterations: int = 50,
    step_size: float = 0.1,
    autograd_mode: str = "vmap",
    benchmarking_costs: bool = False,
    linear_solver_info: Optional[
        Tuple[Type[th.LinearSolver], Type[th.Linearization]]
    ] = None,
    dataset_config: Dict[str, Any] = None,
    parameter_ranges: Dict[str, Any] = None,
) -> List[List[Dict[str, Any]]]:
    verbose = True
    use_gpu = True
    use_cnn = True
    log_params = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "outer_lr": outer_lr,
        "device_param": device_param,
        "max_iterations": max_iterations,
        "step_size": step_size,
        "autograd_mode": autograd_mode,
        "benchmarking_costs": benchmarking_costs,
    }
    log["params"] = log_params
    logger.info(
        "==============================================================="
        "==========================="
    )
    logger.info(f"Batch Size: {batch_size}, " f"Autograd Mode: {autograd_mode}, ")

    logger.info(
        "---------------------------------------------------------------"
        "---------------------------"
    )
    training_sz = -1
    imgH, imgW = dataset_config.get("imgH", 60), dataset_config.get("imgW", 80)
    if dataset_config["name"] == "aerial":
        training_sz = np.max([imgH, imgW])
        imgH, imgW = training_sz, training_sz
    viz_every = dataset_config.get("viz_every", 10)
    save_every = dataset_config.get("save_every", 100)

    viz_dir = os.path.join(os.getcwd(), "viz")
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    id_name = log["sys/id"].fetch()
    viz_dir = os.path.join(viz_dir, id_name)
    os.makedirs(viz_dir, exist_ok=True)

    device = torch.device(f"cuda:{device_param}" if use_gpu else "cpu")
    print(f"Using device: {device}")

    if dataset_config["name"] == "aerial":
        dataset = ImageDataset(
            img_dir=dataset_config["path"],
            training_sz=training_sz,
            param_ranges=parameter_ranges,
            num_samples=dataset_config.get("num_samples", 12800),
            dict_output=True,
        )
    else:
        if dataset_config["path"] is None:
            dataset_paths = prepare_data()
        else:
            dataset_paths = [dataset_config["path"]]
        dataset = HomographyDataset(dataset_paths, imgH, imgW)
    dataset_log = {
        "imgH": imgH,
        "imgW": imgW,
        "training_sz": training_sz,
        "parameter_ranges": parameter_ranges,
        "name": dataset_config["name"],
        "path": dataset_config["path"],
        "num_samples": len(dataset),
    }
    log["dataset"] = dataset_log
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=dataset_config.get("num_workers", 1),
    )
    model_log = {
        "name": model_cfg._target_.split(".")[-1],
        "channels": model_cfg.get("D", 1),
        "blur_type": model_cfg.get("blur_type", "none"),
    }
    log["model"] = model_log

    cnn_model = hydra.utils.instantiate(model_cfg)
    cnn_model.to(device)

    objective = th.Objective()

    data = next(iter(dataloader))
    H8_init = torch.eye(3).reshape(1, 9)[:, :-1].repeat(batch_size, 1)
    feats = torch.zeros_like(data["img1"])
    H8_1_2 = th.Vector(tensor=H8_init, name="H8_1_2")
    feat1 = th.Variable(tensor=feats, name="feat1")
    feat2 = th.Variable(tensor=feats, name="feat2")

    # Set up inner loop optimization.
    homography_cf = th.AutoDiffCostFunction(
        optim_vars=[H8_1_2],
        err_fn=cast(ErrFnType, homography_error_fn),
        dim=1,
        aux_vars=[feat1, feat2],
        autograd_mode=autograd_mode,
    )
    objective.add(homography_cf)

    # Regularization helps avoid crash with using implicit mode.
    reg_w_value = 1e-2
    reg_w = th.ScaleCostWeight(np.sqrt(reg_w_value))
    reg_w.to(dtype=H8_init.dtype)
    vals = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    H8_1_2_id = th.Vector(tensor=vals, name="identity")
    reg_cf = th.Difference(
        H8_1_2, target=H8_1_2_id, cost_weight=reg_w, name="reg_homography"
    )
    objective.add(reg_cf)

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
    theseus_layer = th.TheseusLayer(inner_optim).to(device)

    # Set up outer loop optimization.
    outer_optim = torch.optim.Adam(cnn_model.parameters(), lr=outer_lr)

    itr = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for name, param in cnn_model.named_parameters():
        if param.requires_grad:
            value = 0.0
            log[f"weights_hist/{name}"].append(value)
            log[f"weights_hist/{name}_std"].append(value)
    logger.info(
        "---------------------------------------------------------------"
        "---------------------------"
    )
    cnn_model.train()
    # benchmark_results[i][j] has the results (time/mem) for epoch i and batch j
    benchmark_results: List[List[Dict[str, Any]]] = []
    for epoch in range(num_epochs):
        benchmark_results.append([])
        forward_times: List[float] = []
        forward_mems: List[float] = []
        backward_times: List[float] = []
        backward_mems: List[float] = []
        epoch_loss = 0.0

        for _, data in enumerate(dataloader):
            benchmark_results[-1].append({})
            outer_optim.zero_grad()

            img1 = data["img1"].to(device)
            img2 = data["img2"].to(device)
            img1_norm = normalize_img_batch(img1)
            img2_norm = normalize_img_batch(img2)
            Hgt_1_2 = data["H_1_2"].to(device)

            if use_cnn:  # Use cnn features.
                feat1_tensor = cnn_model.forward(img1_norm)
                feat2_tensor = cnn_model.forward(img2_norm)
            else:  # Use image pixels.
                feat1_tensor = img1
                feat2_tensor = img2

            H8_init = torch.eye(3).reshape(1, 9)[:, :-1].repeat(batch_size, 1)
            H8_init = H8_init.to(device)

            inputs: Dict[str, torch.Tensor] = {
                "H8_1_2": H8_init,
                "feat1": feat1_tensor,
                "feat2": feat2_tensor,
            }
            start_event.record()
            torch.cuda.reset_peak_memory_stats()

            if benchmarking_costs:
                objective.update(inputs)
                inner_optim.linear_solver.linearization.linearize()
            else:
                _, info = theseus_layer.forward(
                    inputs,
                    optimizer_kwargs={
                        "verbose": verbose,
                        "track_err_history": True,
                        "track_state_history": True,
                        "backward_mode": "implicit",
                    },
                )
            end_event.record()
            torch.cuda.synchronize()
            forward_time = start_event.elapsed_time(end_event)
            forward_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            forward_times.append(forward_time)
            forward_mems.append(forward_mem)
            benchmark_results[-1][-1]["ftime"] = forward_time
            benchmark_results[-1][-1]["fmem"] = forward_mem

            if benchmarking_costs:
                continue

            optimizer_info: th.NonlinearOptimizerInfo = cast(
                th.NonlinearOptimizerInfo, info
            )
            err_hist = optimizer_info.err_history
            H_hist = optimizer_info.state_history
            # print("Finished inner loop in %d iters" % len(H_hist))

            Hgt_1_2 = Hgt_1_2.reshape(-1, 9)
            H8_1_2_tensor = theseus_layer.objective.get_optim_var(
                "H8_1_2"
            ).tensor.reshape(-1, 8)
            H_1_2 = torch.cat(
                [H8_1_2_tensor, H8_1_2_tensor.new_ones(H8_1_2_tensor.shape[0], 1)],
                dim=-1,
            )
            # Loss is on four corner error.
            fc_dist = four_corner_dist(
                H_1_2.reshape(-1, 3, 3), Hgt_1_2.reshape(-1, 3, 3), imgH, imgW
            )
            outer_loss = fc_dist.mean()

            start_event.record()
            torch.cuda.reset_peak_memory_stats()
            outer_loss.backward()
            end_event.record()
            torch.cuda.synchronize()
            backward_time = start_event.elapsed_time(end_event)
            backward_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

            backward_times.append(backward_time)
            backward_mems.append(backward_mem)
            benchmark_results[-1][-1]["btime"] = backward_time
            benchmark_results[-1][-1]["bmem"] = backward_mem

            outer_optim.step()
            logger.info(
                "Epoch %d, iteration %d, outer_loss: %.3f"
                % (epoch, itr, outer_loss.item())
            )
            inner_loss_last = err_hist[0][-1].item()
            log["metrics/outer_loss"].append(outer_loss.item())
            log["metrics/inner_loss"].append(inner_loss_last)
            epoch_loss += float(outer_loss.item())
            if itr % viz_every == 0:
                write_gif_batch(viz_dir, feat1_tensor, feat2_tensor, H_hist, Hgt_1_2, err_hist, name=f"feature_homography_{itr}")
                write_gif_batch(viz_dir, img1, img2, H_hist, Hgt_1_2, err_hist, name=f"img_homography_{itr}")
                grad_norm = compute_grad_norm(cnn_model)
                log["metrics/grad_norm"].append(grad_norm)

            if itr % save_every == 0:
                filename = f"last_weights_{id_name}"
                save_path = os.path.join(checkpoint_dir, f"{filename}.ckpt")
                torch.save({"itr": itr, "cnn_model": cnn_model}, save_path)

            itr += 1
        avg_epoch_loss = float(epoch_loss / len(dataloader))
        log["epoch/epoch_loss"].log(avg_epoch_loss)
        logger.info(
            "--------------1-------------------------------------------------"
            "---------------------------"
        )
        log["performance/forward_time_ms"].extend(forward_times)
        log["performance/forward_memory_MB"].extend(forward_mems)
        log["performance/backward_time_ms"].extend(backward_times)
        log["performance/backward_memory_MB"].extend(backward_mems)
        logger.info(f"Forward pass took {sum(forward_times)} ms/epoch.")
        logger.info(f"Forward pass took {sum(forward_mems)/len(forward_mems)} MBs.")
        log["performance/forward_time_total_ms"].append(sum(forward_times))
        log["performance/forward_memory_avg_MB"].append(sum(forward_mems) / len(forward_mems))
        # logger.info(f"benchmarking_costs: {benchmarking_costs}")
        if not benchmarking_costs:
            logger.info(f"Backward pass took {sum(backward_times)} ms/epoch.")
            logger.info(
                f"Backward pass took {sum(backward_mems)/len(backward_mems)} MBs."
            )
            log["performance/backward_time_total_ms"].append(sum(backward_times))
            log["performance/backward_memory_avg_MB"].append(
                sum(backward_mems) / len(backward_mems)
            )
        logger.info(
            "---------------------------------------------------------------"
            "---------------------------"
        )
        for name, param in cnn_model.named_parameters():
            if param.requires_grad:
                value = param.detach().cpu().numpy()
                log[f"weights_hist/{name}"].append(value.mean())
                log[f"weights_hist/{name}_std"].append(value.std())

    return benchmark_results


@hydra.main(config_path="./configs/", config_name="homography_estimation")
def main(cfg):
    log = neptune.init_run(
        project="maciej.krupka/gps-denied",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NDk0MTVlYy1lZDE4LTQxNzEtYjNkNC1hMjkzOWRjMTU4YTAifQ==",
    )
    benchmark_results = run(
        log,
        model_cfg=cfg.model,
        batch_size=cfg.outer_optim.batch_size,
        outer_lr=cfg.outer_optim.lr,
        device_param=cfg.outer_optim.device,
        num_epochs=cfg.outer_optim.num_epochs,
        max_iterations=cfg.inner_optim.max_iters,
        step_size=cfg.inner_optim.step_size,
        autograd_mode=cfg.autograd_mode,
        benchmarking_costs=cfg.benchmarking_costs,
        dataset_config=cfg.dataset,
        parameter_ranges=cfg.parameter_ranges,
        linear_solver_info=cfg.get("linear_solver_info", None),
    )
    id_name = log["sys/id"].fetch()
    torch.save(
        benchmark_results, pathlib.Path(os.getcwd()) / f"benchmark_results_{id_name}.pt"
    )
    log.stop()


if __name__ == "__main__":
    torch.manual_seed(0)

    main()
