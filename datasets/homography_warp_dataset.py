import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import random
from math import sin, cos, radians
from pathlib import Path
from core.geometry import (
    param_to_H,
    H_to_param,
    warp_hmg,
    param_to_A,
    warp_perspective_norm,
)
from theseus.third_party.easyaug import RandomPhotoAug
import kornia
import numpy as np

random.seed(42)


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
        param_ranges: dict,
        num_samples: int = 10_000,
        same_pair: bool = False,
        photo_aug: bool = False,
    ):
        super().__init__()
        self.image_paths = glob.glob(str(Path(img_dir) / "*.png"))
        assert len(self.image_paths) >= 2, "Need at least two images in the folder"
        # if you have anouth ram
        self.images = [Image.open(p).convert("RGB") for p in self.image_paths]
        # sizes
        self.training_sz = training_sz
        self.num_samples = num_samples  # __len__
        self.same_pair = same_pair  # if True, use the same image for both inputs
        # augment parameter ranges
        self.lower_sz = param_ranges["lower_sz"]
        self.upper_sz = param_ranges["upper_sz"]
        self.warp_pad = param_ranges["warp_pad"]
        self.min_scale = param_ranges["min_scale"]
        self.max_scale = param_ranges["max_scale"]
        self.angle_range = param_ranges["angle_range"]
        self.projective_range = param_ranges["projective_range"]
        self.translation_range = param_ranges["translation_range"]
        self.training_sz_pad = round(training_sz + training_sz * 2 * self.warp_pad)
        self.pad_side = round(self.training_sz * self.warp_pad)

        self.photo_aug = photo_aug
        if self.photo_aug:
            self.rpa = RandomPhotoAug()
            prob = 0.2  # Probability of augmentation applied.
            mag = 0.2  # Magnitude of augmentation [0: none, 1: max]
            self.rpa.set_all_probs(prob)
            self.rpa.set_all_mags(mag)

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self.num_samples

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def __getitem__(self, idx: int):
        """Generate ONE synthetic training triple."""
        assert 0 <= idx < self.num_samples, "Index out of bounds"
        pair = 2
        if self.same_pair:
            pair = 1

        images = random.sample(self.images, pair)
        img = images[0].copy()
        template = images[pair - 1].copy()

        in_W, in_H = img.size

        # ---------------- random crop (with padding) ------------------ #
        seg_sz = random.randint(self.lower_sz, self.upper_sz)
        seg_sz_pad = round(seg_sz + seg_sz * 2 * self.warp_pad)

        loc_x = random.randint(0, in_W - seg_sz_pad - 1)
        loc_y = random.randint(0, in_H - seg_sz_pad - 1)

        # enlarged crop (still PIL)
        img_crop = img.crop(
            (loc_x, loc_y, loc_x + seg_sz_pad, loc_y + seg_sz_pad)
        ).resize((self.training_sz_pad, self.training_sz_pad))

        template_crop = template.crop(
            (loc_x, loc_y, loc_x + seg_sz_pad, loc_y + seg_sz_pad)
        ).resize((self.training_sz_pad, self.training_sz_pad))

        # to tensor  ---------------------------------------------------- #
        img_crop = np.array(img_crop).astype(np.float32)
        img_tensor = torch.from_numpy(img_crop / 255.0).permute(2, 0, 1)[None]

        template_crop = np.asarray(template_crop).astype(np.float32)
        template_image = torch.from_numpy(template_crop / 255.0).permute(2, 0, 1)[None]
        # ----------------- random ground-truth params ------------------ #
        scale = random.uniform(self.min_scale, self.max_scale)
        angle = random.uniform(-self.angle_range, self.angle_range)
        trans_x = random.uniform(-self.translation_range, self.translation_range)
        trans_y = random.uniform(-self.translation_range, self.translation_range)
        rad_ang = radians(angle)

        params = torch.tensor([[scale, rad_ang, trans_x, trans_y]], dtype=torch.float32)
        H_gt = kornia.geometry.convert_affinematrix_to_homography(param_to_A(params))

        # apply random photometric augmentations
        if self.photo_aug:
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            template_image = torch.clamp(template_image, 0.0, 1.0)
            img_tensor = self.rpa.forward(img_tensor)
            template_image = self.rpa.forward(template_image)

        return template_image.squeeze(0), img_tensor.squeeze(0), H_gt

    def get_collate_fn(self):
        crop = slice(self.pad_side, self.pad_side + self.training_sz)

        def collate_fn(batch):
            """
            Custom collate function to handle the dataset output.
            """
            imgs, templates, Hgs = zip(*batch)
            imgs = torch.stack(imgs)
            templates = torch.stack(templates)
            Hgs = torch.stack(Hgs)
            # # ----------------- apply inverse warp to image ----------------- #
            # #  → we want 'warped_image' such that template == warp(warped_image, p_gt)
            warped_image = warp_perspective_norm(Hgs, templates)
            # ------------- remove padding to reach final size -------------- #
            warped_image = warped_image[:, :, crop, crop]
            imgs = imgs[:, :, crop, crop]
            return {
                "img1": imgs,
                "img2": warped_image,
                "H_1_2": Hgs,
            }

        return collate_fn

    def update_parameter_ranges(self, new_ranges: dict) -> bool:
        """
        Updates the dataset's parameter ranges from a dictionary.

        Args:
            new_ranges (dict): A dictionary containing the new parameter ranges.
                               Keys should match the class attribute names.

        Returns:
            bool: True if any parameter value was changed, False otherwise.
        """
        changed = False
        params_to_update = [
            "lower_sz",
            "upper_sz",
            "warp_pad",
            "min_scale",
            "max_scale",
            "angle_range",
            "projective_range",
            "translation_range",
        ]

        for param in params_to_update:
            if param in new_ranges:
                current_value = getattr(self, param)
                new_value = new_ranges[param]
                if current_value != new_value:
                    setattr(self, param, new_value)
                    changed = True

        # If warp_pad was changed, we must recalculate dependent parameters
        if changed and "warp_pad" in new_ranges:
            # Check if warp_pad specifically was the reason for the change,
            # though recalculating every time `changed` is true is also safe.
            self.training_sz_pad = round(
                self.training_sz + self.training_sz * 2 * self.warp_pad
            )
            self.pad_side = round(self.training_sz * self.warp_pad)

        return changed

    def get_parameter_ranges(self) -> dict:
        """
        Returns the current parameter ranges as a dictionary.
        """
        return {
            "lower_sz": self.lower_sz,
            "upper_sz": self.upper_sz,
            "warp_pad": self.warp_pad,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "angle_range": self.angle_range,
            "projective_range": self.projective_range,
            "translation_range": self.translation_range,
        }
