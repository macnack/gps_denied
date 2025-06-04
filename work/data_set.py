import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import random
from math import sin, cos, pi, radians
import model as dlk
from pathlib import Path


class ImageDataset(Dataset):
    """
    Returns (warped_image, template_image, p_gt) for each index.

    * warped_image  : Tensor [3, training_sz, training_sz]
    * template_image: Tensor [3, training_sz, training_sz]
    * p_gt          : Tensor [8, 1]   -- ground-truth warp parameters
    """
    def __init__(
        self,
        img_dir: str | Path,
        training_sz: int,
        training_sz_pad: int,
        param_ranges: dict,
        num_samples: int = 10_000,
        transform: transforms.Compose | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.image_paths = glob.glob(str(Path(img_dir) / "*.png"))
        assert len(self.image_paths) >= 1, "Need at least one image in the folder"

        # sizes
        self.training_sz      = training_sz
        self.training_sz_pad  = training_sz_pad
        self.num_samples      = num_samples          # __len__

        # augment parameter ranges
        self.lower_sz         = param_ranges["lower_sz"]
        self.upper_sz         = param_ranges["upper_sz"]
        self.warp_pad         = param_ranges["warp_pad"]
        self.min_scale        = param_ranges["min_scale"]
        self.max_scale        = param_ranges["max_scale"]
        self.angle_range      = param_ranges["angle_range"]
        self.projective_range = param_ranges["projective_range"]
        self.translation_range= param_ranges["translation_range"]

        # misc
        self.transform = transform or transforms.ToTensor()
        self.device    = torch.device(device)

        # reusable helpers
        self.inverse   = dlk.InverseBatch()          # same object for every call

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self.num_samples

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def __getitem__(self, idx: int):
        """Generate ONE synthetic training triple."""
        img_path      = random.choice(self.image_paths)

        # Use **one** source image, keep a *copy* as template, warp the other
        img      = Image.open(img_path)
        template = img.copy()

        in_W, in_H = img.size

        # ---------------- random crop (with padding) ------------------ #
        seg_sz     = random.randint(self.lower_sz, self.upper_sz)
        seg_sz_pad = round(seg_sz + seg_sz * 2 * self.warp_pad)

        loc_x = random.randint(0, in_W - seg_sz_pad - 1)
        loc_y = random.randint(0, in_H - seg_sz_pad - 1)

        # enlarged crop (still PIL)
        img_crop = img.crop((loc_x, loc_y,
                             loc_x + seg_sz_pad,
                             loc_y + seg_sz_pad)).resize(
                                (self.training_sz_pad, self.training_sz_pad)
                             )

        template_crop = template.crop((loc_x, loc_y,
                                       loc_x + seg_sz_pad,
                                       loc_y + seg_sz_pad)).resize(
                                          (self.training_sz_pad, self.training_sz_pad)
                                       )

        # to tensor  ---------------------------------------------------- #
        img_tensor      = self.transform(img_crop)         # [3, H_pad, W_pad]
        template_tensor = self.transform(template_crop)    # ditto

        # ----------------- random ground-truth params ------------------ #
        scale         = random.uniform(self.min_scale, self.max_scale)
        angle         = random.uniform(-self.angle_range, self.angle_range)
        proj_x        = random.uniform(-self.projective_range, self.projective_range)
        proj_y        = random.uniform(-self.projective_range, self.projective_range)
        trans_x       = random.uniform(-self.translation_range, self.translation_range)
        trans_y       = random.uniform(-self.translation_range, self.translation_range)

        rad_ang = radians(angle)
        p_gt = torch.tensor(
            [
                scale + cos(rad_ang) - 2,
               -sin(rad_ang),
                trans_x,
                sin(rad_ang),
                scale + cos(rad_ang) - 2,
                trans_y,
                proj_x,
                proj_y,
            ],
            dtype=torch.float32,
            device=self.device,
        ).view(8, 1)                                     # [8,1]

        # ----------------- apply inverse warp to image ----------------- #
        #  → we want 'img_tensor_w' such that template == warp(img_tensor_w, p_gt)
        img_tensor_4d = img_tensor.unsqueeze(0).to(self.device)         # [1,3,H,W]
        H             = dlk.param_to_H(p_gt.unsqueeze(0))               # [1,3,3]
        H_inv         = self.inverse(H)
        img_w, _      = dlk.warp_hmg(img_tensor_4d, dlk.H_to_param(H_inv))
        img_tensor_w  = img_w.squeeze(0)                                # back to [3,H,W]

        # ------------- remove padding to reach final size -------------- #
        pad_side = round(self.training_sz * self.warp_pad)
        img_tensor_w  = img_tensor_w[:, pad_side : pad_side + self.training_sz,
                                        pad_side : pad_side + self.training_sz]
        template_tensor = template_tensor[:, pad_side : pad_side + self.training_sz,
                                              pad_side : pad_side + self.training_sz]

        # make sure both live on the requested device
        template_tensor = template_tensor.to(self.device)

        return img_tensor_w, template_tensor, p_gt

from torch.utils.data import DataLoader
if __name__ == "__main__":
    param_ranges = {
        'lower_sz': 80,
        'upper_sz': 120,
        'warp_pad': 0.2,
        'min_scale': 0.8,
        'max_scale': 1.2,
        'angle_range': 30,
        'projective_range': 0.001,
        'translation_range': 5
    }

    dataset = ImageDataset(
        img_dir='/home/user/work/sat_data/woodbridge/images',
        training_sz=100,
        training_sz_pad=120,
        param_ranges=param_ranges,
        transform=transforms.ToTensor(),
        device='cuda'  # or 'cpu'
    )

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    # how to check if the dataloader works
    for img_batch, template_batch, param_batch in dataloader:
        print("Image batch shape:", img_batch.shape)
        print("Template batch shape:", template_batch.shape)
        print("Parameter batch shape:", param_batch.shape)
        break  # remove this to iterate through the entire dataset
