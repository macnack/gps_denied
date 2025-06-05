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
USE_CUDA = torch.cuda.is_available()
# size scale range
min_scale = 0.75
max_scale = 1.25

# rotation range (-angle_range, angle_range)
angle_range = 15 # degrees

# projective variables (p7, p8)
projective_range = 0

# translation (p3, p6)
translation_range = 10 # pixels

# possible segment sizes
lower_sz = 200 # pixels, square
upper_sz = 220

# amount to pad when cropping segment, as ratio of size, on all 4 sides
warp_pad = 0.4

# normalized size of all training pairs
training_sz = 175
training_sz_pad = round(training_sz + training_sz * 2 * warp_pad)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("MODE")
	parser.add_argument("FOLDER_NAME")
	parser.add_argument("DATAPATH")
	parser.add_argument("MODEL_PATH")
	parser.add_argument("VGG_MODEL_PATH")
	parser.add_argument("-t","--TEST_DATA_SAVE_PATH")

	args = parser.parse_args()

	MODE = args.MODE
	FOLDER_NAME = args.FOLDER_NAME
	FOLDER = FOLDER_NAME + '/'
	DATAPATH = args.DATAPATH
	MODEL_PATH = args.MODEL_PATH
	VGG_MODEL_PATH = args.VGG_MODEL_PATH

	if MODE == 'test':
		if args.TEST_DATA_SAVE_PATH == None:
			exit('Must supply TEST_DATA_SAVE_PATH argument in test mode')
		else:
			TEST_DATA_SAVE_PATH = args.TEST_DATA_SAVE_PATH

def train():
    device = torch.device('cuda' if USE_CUDA else 'cpu')
    run = neptune.init_run(
    project="maciej.krupka/gps-denied",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NDk0MTVlYy1lZDE4LTQxNzEtYjNkNC1hMjkzOWRjMTU4YTAifQ==",
    )  # your credentials

    # Initialize model
    dlk_net = dlk.DeepLK(dlk.vgg16Conv(VGG_MODEL_PATH)).to(device)
    # summary(dlk_net, input_size=[(1, 3, 128, 128), (1, 3, 128, 128)])

    lr = 0.0001
    epoch = 10
    batch_size = 1
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, dlk_net.parameters()), lr=lr)
    run["parameters"] = {
    "min_scale": min_scale,
    "max_scale": max_scale,
    "angle_range": angle_range,
    "projective_range": projective_range,
    "translation_range": translation_range,
    "training_sz": training_sz,
    "training_sz_pad": training_sz_pad,
    "lr": lr,
    "epochs": epoch,
    "batch_size" : batch_size,
    }
    # Dataset and DataLoader setup
    param_ranges = {
        'lower_sz': lower_sz,
        'upper_sz': upper_sz,
        'warp_pad': warp_pad,
        'min_scale': min_scale,
        'max_scale': max_scale,
        'angle_range': angle_range,
        'projective_range': projective_range,
        'translation_range': translation_range,
    }

    transform = transforms.ToTensor()
    dataset = ImageDataset(
        img_dir=DATAPATH + FOLDER + '/images',
        training_sz=training_sz,
        training_sz_pad=training_sz_pad,
        param_ranges=param_ranges,
        num_samples =100,
        transform=transform,
        device=device
    )

    valid_dataset = ImageDataset(
        img_dir=DATAPATH + FOLDER + '/images',
        training_sz=training_sz,
        training_sz_pad=training_sz_pad,
        param_ranges=param_ranges,
        num_samples =10,
        transform=transform,
        device=device
    )

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # train loader size
    print(len(dataset))
    best_valid_loss = float('inf')

    print('Training...')
    for epoch in range(1, epoch):  # small number of epochs, increase if needed
        dlk_net.train()
        for batch_idx, (img_batch, template_batch, param_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            img_batch = img_batch.to(device)
            template_batch = template_batch.to(device)
            param_batch = param_batch.to(device)

            img_batch = dlk.normalize_img_batch(img_batch)
            template_batch = dlk.normalize_img_batch(template_batch)

            pred_params, _ = dlk_net(img_batch, template_batch, tol=1e-3, max_itr=1, conv_flag=1)
            loss = dlk.corner_loss(pred_params, param_batch, training_sz_pad)
            run["train/loss"].log(loss.item())

            loss.backward()
            
            # total_norm = 0
            # for name, param in dlk_net.named_parameters():
                # print(name, param.grad is None, param.grad.norm() if param.grad is not None else 0)
            # for p in dlk_net.parameters():
            #     param_norm = p.grad.data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** (1. / 2)
            # run["train/grad_norm"].log(total_norm)
            # run["train/learning_rate"].log(optimizer.param_groups[0]['lr'])
            optimizer.step()
            
            total_weight_norm = 0.0
            for p in dlk_net.parameters():
                param_norm = p.data.norm(2)
                total_weight_norm += param_norm.item() ** 2
            total_weight_norm = total_weight_norm ** 0.5
            run["train/weight_norm"].log(total_weight_norm)
        
            if batch_idx % batch_size == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Training Loss: {loss.item():.6f}")

        # Validation
        dlk_net.eval()
        total_val_loss = 0
        with torch.no_grad():
            for img_batch, template_batch, param_batch in valid_loader:
                img_batch = img_batch.to(device)
                template_batch = template_batch.to(device)
                param_batch = param_batch.to(device)

                img_batch = dlk.normalize_img_batch(img_batch)
                template_batch = dlk.normalize_img_batch(template_batch)

                pred_params, _ = dlk_net(img_batch, template_batch, tol=1e-3, max_itr=1, conv_flag=1)
                val_loss = dlk.corner_loss(pred_params, param_batch, training_sz_pad)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(valid_loader)
        run["val/loss"].log(avg_val_loss)

        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            torch.save(dlk_net.conv_func, MODEL_PATH)
            print("New best model saved.")

        gc.collect()
        
        
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    print('PID: ', os.getpid())
    train()