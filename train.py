from models import DeepLK, vgg16Conv
import torch
import torch.optim as optim
import torch.nn as nn
from torch import no_grad, save
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import ImageDataset
from core import normalize_img_batch, corner_loss
from math import sqrt
import os
import gc
VGG_MODEL_PATH = None
DATAPATH = "./sat_data/"
FOLDER = "woodbridge/"
train_samples = 512
valid_samples = 128
DEBUG = True
MODEL_PATH = "."
def log_batch_loss(epoch, batch_idx, loss_val):
    print(
        f"[cyan]Epoch {epoch}, Batch {batch_idx}[/cyan] - [bold green]Training Loss:[/] {loss_val:.3f}"
    )


def log_epoch_loss(epoch, avg_loss):
    print(
        f"[magenta]Epoch {epoch}[/magenta] - [bold yellow]Average Training Loss:[/] {avg_loss:.6f}"
    )
    
def train(run, device, config):

    # Initialize model
    dlk_net = DeepLK(vgg16Conv(VGG_MODEL_PATH)).to(device)
    run["model/architecture"] = str(dlk_net)
    run["model/parameters"] = sum(p.numel() for p in dlk_net.parameters())
    # summary(dlk_net, input_size=[(1, 3, 128, 128), (1, 3, 128, 128)])

    lr = 0.0001
    num_epoch = 10
    batch_size = 2
    gradiend_cliping_norm = 0.5
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, dlk_net.parameters()), lr=lr
    )
    parameter_ranges = config['parameter_ranges']
    training_sz_pad = round(config["training_sz"] + config["training_sz"] * 2 * parameter_ranges["warp_pad"])
    run["parameters"] = {
        "lower_sz": parameter_ranges["lower_sz"],
        "upper_sz": parameter_ranges["upper_sz"],
        "warp_pad": parameter_ranges["warp_pad"],
        "min_scale": parameter_ranges["min_scale"],
        "max_scale": parameter_ranges["max_scale"],
        "angle_range": parameter_ranges["angle_range"],
        "projective_range": parameter_ranges["projective_range"],
        "translation_range": parameter_ranges["translation_range"],
        "training_sz": config["training_sz"],
        "training_sz_pad": training_sz_pad,
        "lr": config["lr"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "gradient_clipping_norm": config["gradient_clipping_norm"],
    }
    # Dataset and DataLoader setup
    transform = transforms.ToTensor()
    dataset = ImageDataset(
        img_dir=DATAPATH + FOLDER + "/images",
        training_sz=config["training_sz"],
        param_ranges=parameter_ranges,
        num_samples=train_samples,
        transform=transform,
    )

    valid_dataset = ImageDataset(
        img_dir=DATAPATH + FOLDER + "/images",
        training_sz=config["training_sz"],
        param_ranges=parameter_ranges,
        num_samples=valid_samples,
        transform=transform,
    )
    print("Dataloading complete.")
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=config["num_workers"]
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["val_workers"],
    )
    run["parameters/train_dataset_size"] = len(dataset)
    run["parameters/train_workers"] = config["num_workers"]
    run["parameters/valid_dataset_size"] = len(valid_dataset)
    run["parameters/valid_workers"] = config["val_workers"]
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

            img_batch = normalize_img_batch(img_batch)
            template_batch = normalize_img_batch(template_batch)

            pred_params, _ = dlk_net(
                img_batch, template_batch, tol=1e-3, max_itr=1, conv_flag=1
            )
            loss = corner_loss(pred_params, param_batch, training_sz_pad)
            norm_loss = loss.item() / float(batch_size)
            run["train/loss"].log(norm_loss)
            epoch_loss += norm_loss
            loss.backward()
            nn.utils.clip_grad_norm_(
                dlk_net.parameters(), max_norm=config["gradient_clipping_norm"]
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
        with no_grad():
            for img_batch, template_batch, param_batch in valid_loader:
                img_batch = img_batch.to(device)
                template_batch = template_batch.to(device)
                param_batch = param_batch.to(device)

                img_batch = normalize_img_batch(img_batch)
                template_batch = normalize_img_batch(template_batch)

                pred_params, _ = dlk_net(
                    img_batch, template_batch, tol=1e-3, max_itr=1, conv_flag=1
                )
                val_loss = corner_loss(pred_params, param_batch, training_sz_pad)
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
            save(dlk_net.conv_func, filepath)
            print(f"New best model saved as {filepath}")
        gc.collect()

from torch import device
import neptune
import multiprocessing as mp
lower_sz = 200  # pixels, square
upper_sz = 220
warp_pad = 0.4
training_sz = 175
min_scale = 0.75
max_scale = 1.25
angle_range = 15  # degrees
projective_range = 0
translation_range = 10  # pixels

parameters = {
    "parameter_ranges": {
        "lower_sz": lower_sz,
        "upper_sz": upper_sz,
        "warp_pad": warp_pad,
        "min_scale": min_scale,
        "max_scale": max_scale,
        "angle_range": angle_range,
        "projective_range": projective_range,
        "translation_range": translation_range,
    },
    "training_sz": training_sz,
    "lr": 0.0001,
    "epochs": 10,
    "batch_size": 2,
    "gradient_clipping_norm": 0.5,
    "num_workers": 1,
    "val_workers": 1,
}
USE_CUDA = torch.cuda.is_available()
if __name__ == "__main__":
    device_ = device(f"cuda:{0}" if USE_CUDA else "cpu")
    run = neptune.init_run(
        project="maciej.krupka/gps-denied",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NDk0MTVlYy1lZDE4LTQxNzEtYjNkNC1hMjkzOWRjMTU4YTAifQ==",
    )  # your credentials
    mp.set_start_method("spawn", force=True)
    print("PID: ", os.getpid())
    print(f"Run ID: {run['sys/id'].fetch()}")
    train(run=run, device=device_, config=parameters)