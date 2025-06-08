from torch.utils.data import DataLoader
from torchvision import transforms
from data_set import ImageDataset
import torch
import argparse
import torch.optim as optim
import model as dlk
import gc
import os
import neptune
import torch.multiprocessing as mp
from torchinfo import summary
from rich import print
import time
from datetime import datetime
import torch.nn as nn
from math import sqrt
import numpy as np

DEBUG = True
USE_CUDA = torch.cuda.is_available()
# USE_CUDA = False
train_samples = 512
valid_samples = 128
num_workers = 0
validation_workers = num_workers
# size scale range
min_scale = 0.75
max_scale = 1.25

# rotation range (-angle_range, angle_range)
angle_range = 15  # degrees

# projective variables (p7, p8)
projective_range = 0

# translation (p3, p6)
translation_range = 10  # pixels

# possible segment sizes
lower_sz = 200  # pixels, square
upper_sz = 220

# amount to pad when cropping segment, as ratio of size, on all 4 sides
warp_pad = 0.4

# normalized size of all training pairs
training_sz = 175
training_sz_pad = round(training_sz + training_sz * 2 * warp_pad)

param_ranges = {
    "lower_sz": lower_sz,
    "upper_sz": upper_sz,
    "warp_pad": warp_pad,
    "min_scale": min_scale,
    "max_scale": max_scale,
    "angle_range": angle_range,
    "projective_range": projective_range,
    "translation_range": translation_range,
}


def log_batch_loss(epoch, batch_idx, loss_val):
    print(
        f"[cyan]Epoch {epoch}, Batch {batch_idx}[/cyan] - [bold green]Training Loss:[/] {loss_val:.3f}"
    )


def log_epoch_loss(epoch, avg_loss):
    print(
        f"[magenta]Epoch {epoch}[/magenta] - [bold yellow]Average Training Loss:[/] {avg_loss:.6f}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("MODE")
    parser.add_argument("FOLDER_NAME")
    parser.add_argument("DATAPATH")
    parser.add_argument(
        "--MODEL_DIRECTORY",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--VGG_MODEL_PATH",
        type=str,
        default="x",
        help="Path to VGG model weights, or 'x' to skip VGG initialization",
    )
    parser.add_argument(
        "-t",
        "--TEST_DATA_SAVE_PATH",
        type=str,
        default="results",
        help="Path to save test data results",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device number to use (default: 0)"
    )
    args = parser.parse_args()

    MODE = args.MODE
    FOLDER_NAME = args.FOLDER_NAME
    FOLDER = FOLDER_NAME + "/"
    DATAPATH = args.DATAPATH
    MODEL_PATH = args.MODEL_DIRECTORY
    os.makedirs(MODEL_PATH, exist_ok=True)
    VGG_MODEL_PATH = args.VGG_MODEL_PATH
    DEVICE = args.device
    if VGG_MODEL_PATH == "x":
        VGG_MODEL_PATH = None
    TEST_DATA_SAVE_PATH = args.TEST_DATA_SAVE_PATH


def train(run, device):

    # Initialize model
    dlk_net = dlk.DeepLK(dlk.vgg16Conv(VGG_MODEL_PATH)).to(device)
    run["model/architecture"] = str(dlk_net)
    run["model/parameters"] = sum(p.numel() for p in dlk_net.parameters())
    # summary(dlk_net, input_size=[(1, 3, 128, 128), (1, 3, 128, 128)])

    lr = 0.0001
    num_epoch = 10
    batch_size = 2
    gradiend_cliping_norm = 1.0
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, dlk_net.parameters()), lr=lr
    )
    run["parameters"] = {
        "min_scale": min_scale,
        "max_scale": max_scale,
        "angle_range": angle_range,
        "projective_range": projective_range,
        "translation_range": translation_range,
        "training_sz": training_sz,
        "training_sz_pad": training_sz_pad,
        "lr": lr,
        "epochs": num_epoch,
        "batch_size": batch_size,
        "gradient_clipping_norm": gradiend_cliping_norm,
    }
    # Dataset and DataLoader setup
    transform = transforms.ToTensor()
    dataset = ImageDataset(
        img_dir=DATAPATH + FOLDER + "/images",
        training_sz=training_sz,
        training_sz_pad=training_sz_pad,
        param_ranges=param_ranges,
        num_samples=train_samples,
        transform=transform,
        device=device,
    )

    valid_dataset = ImageDataset(
        img_dir=DATAPATH + FOLDER + "/images",
        training_sz=training_sz,
        training_sz_pad=training_sz_pad,
        param_ranges=param_ranges,
        num_samples=valid_samples,
        transform=transform,
        device=device,
    )
    print("Dataloading complete.")
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=validation_workers,
    )
    run["parameters/train_dataset_size"] = len(dataset)
    run["parameters/train_workers"] = num_workers
    run["parameters/valid_dataset_size"] = len(valid_dataset)
    run["parameters/valid_workers"] = validation_workers
    best_valid_loss = float("inf")

    print("Training...")
    for epoch in range(num_epoch):  # small number of epochs, increase if needed
        dlk_net.train()
        epoch_loss = 0.0
        for batch_idx, (img_batch, template_batch, param_batch) in enumerate(
            train_loader
        ):
            # start_time = time.time()
            optimizer.zero_grad()
            img_batch = img_batch.to(device)
            template_batch = template_batch.to(device)
            param_batch = param_batch.to(device)

            img_batch = dlk.normalize_img_batch(img_batch)
            template_batch = dlk.normalize_img_batch(template_batch)

            pred_params, _ = dlk_net(
                img_batch, template_batch, tol=1e-3, max_itr=50, conv_flag=1
            )
            loss = dlk.corner_loss(pred_params, param_batch, training_sz_pad)
            norm_loss = loss.item() / float(batch_size)
            run["train/loss"].log(norm_loss)
            epoch_loss += norm_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                dlk_net.parameters(), max_norm=gradiend_cliping_norm
            )  # Gradient clipping
            optimizer.step()
            # elapsed = time.time() - start_time
            # run["train/time_per_batch"].log(elapsed)
            if batch_idx % (batch_size) == 0:
                total_weight_norm = 0.0
                for p in dlk_net.parameters():
                    param_norm = p.data.norm(2)
                    total_weight_norm += param_norm.item() ** 2
                total_weight_norm = total_weight_norm**0.5
                run["train/weight_norm"].log(total_weight_norm)
                if DEBUG:
                    log_batch_loss(epoch, batch_idx, norm_loss)

        avg_epoch_loss = epoch_loss / len(train_loader)
        run["train/epoch_loss"].log(avg_epoch_loss)
        run["train/epoch"] = epoch
        log_epoch_loss(epoch, avg_epoch_loss)
        # Validation
        # should i empty cache here?
        # torch.cuda.empty_cache()
        dlk_net.eval()
        total_val_loss = 0
        with torch.no_grad():
            for img_batch, template_batch, param_batch in valid_loader:
                img_batch = img_batch.to(device)
                template_batch = template_batch.to(device)
                param_batch = param_batch.to(device)

                img_batch = dlk.normalize_img_batch(img_batch)
                template_batch = dlk.normalize_img_batch(template_batch)

                pred_params, _ = dlk_net(
                    img_batch, template_batch, tol=1e-3, max_itr=1, conv_flag=1
                )
                val_loss = dlk.corner_loss(pred_params, param_batch, training_sz_pad)
                rmse = sqrt(val_loss.item() / 4)
                run["val/rmse"].log(rmse)
                total_val_loss += val_loss.item() / float(batch_size)

        avg_val_loss = total_val_loss / len(valid_loader)
        run["val/loss"].log(avg_val_loss)

        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            run_name = run["sys/id"].fetch()
            filename = f"dlk_weights_{run_name}.pt"
            filepath = os.path.join(MODEL_PATH, filename)
            torch.save(dlk_net.conv_func, filepath)
            print(f"New best model saved as {filepath}")
        gc.collect()


def test(run, device, test_rounds=50):
    os.makedirs(TEST_DATA_SAVE_PATH, exist_ok=True)

    device = torch.device("cuda" if USE_CUDA else "cpu")

    dlk_vgg16 = dlk.DeepLK(dlk.vgg16Conv(VGG_MODEL_PATH)).to(device)
    dlk_trained = dlk.DeepLK(dlk.custom_net(MODEL_PATH)).to(device)

    transform = transforms.ToTensor()
    test_ds = ImageDataset(
        DATAPATH + FOLDER + "images",
        training_sz,
        training_sz_pad,
        param_ranges,
        transform,
        device,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    results = np.zeros((test_rounds, 5), dtype=float)

    with torch.no_grad():
        for i, (img, tmpl, p_gt) in enumerate(test_loader):
            if i >= test_rounds:
                break
            img, tmpl, p_gt = img.to(device), tmpl.to(device), p_gt.to(device)

            img_n = dlk.normalize_img_batch(img)
            tmpl_n = dlk.normalize_img_batch(tmpl)

            img_c = nn.AvgPool2d(4)(img_n)
            tmpl_c = nn.AvgPool2d(4)(tmpl_n)

            vgg_p, _ = dlk_vgg16(img_n, tmpl_n, tol=1e-3, max_itr=70, conv_flag=1)
            trained_p, _ = dlk_trained(img_n, tmpl_n, tol=1e-3, max_itr=70, conv_flag=1)
            coarse_p, _ = dlk_vgg16(img_c, tmpl_c, tol=1e-3, max_itr=70, conv_flag=0)

            zero_p = torch.zeros_like(p_gt)

            vgg_loss = dlk.corner_loss(vgg_p, p_gt, training_sz_pad)
            trained_loss = dlk.corner_loss(trained_p, p_gt, training_sz_pad)
            coarse_loss = dlk.corner_loss(coarse_p, p_gt, training_sz_pad)
            no_op_loss = dlk.corner_loss(zero_p, p_gt, training_sz_pad)

            run["test/vgg_loss"].log(vgg_loss)
            run["test/trained_loss"].log(trained_loss)
            run["test/coarse_loss"].log(coarse_loss)
            run["test/noop_loss"].log(no_op_loss)
            results[i, :4] = [
                vgg_loss.item(),
                trained_loss.item(),
                coarse_loss.item(),
                no_op_loss.item(),
            ]
            print(
                f"Test {i+1}/{test_rounds}:  vgg={sqrt(vgg_loss.item()/4):.2f}  "
                f"trained={sqrt(trained_loss.item()/4):.2f}  coarse={sqrt(coarse_loss.item()/4):.2f}  "
                f"no-op={sqrt(no_op_loss.item()/4):.2f}"
            )
    TEST_DATA_SAVE_PATH = os.path.join(
        TEST_DATA_SAVE_PATH,
        f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    np.savetxt(TEST_DATA_SAVE_PATH, results, delimiter=",")
    run["test/csv"].upload(TEST_DATA_SAVE_PATH)

    print(f"Saved results to {TEST_DATA_SAVE_PATH}")


if __name__ == "__main__":
    device = torch.device(f"cuda:{DEVICE}" if USE_CUDA else "cpu")
    run = neptune.init_run(
        project="maciej.krupka/gps-denied",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NDk0MTVlYy1lZDE4LTQxNzEtYjNkNC1hMjkzOWRjMTU4YTAifQ==",
    )  # your credentials
    mp.set_start_method("spawn", force=True)
    print("PID: ", os.getpid())
    print(f"Run ID: {run['sys/id'].fetch()}")

    if MODE == "train":
        print(f"Training mode: {FOLDER_NAME}")
        train(run=run, device=device)
