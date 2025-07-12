import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import random
from math import sin, cos, radians
from pathlib import Path
from core.geometry import param_to_H, H_to_param, warp_hmg
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
        transform: transforms.Compose | None = None,
        dict_output: bool = False,
        same_pair: bool = False,
    ):
        super().__init__()
        self.image_paths = glob.glob(str(Path(img_dir) / "*.png"))
        assert len(self.image_paths) >= 2, "Need at least two images in the folder"
        # if you have anouth ram
        self.images = [Image.open(p).convert("RGB") for p in self.image_paths]
        # sizes
        self.training_sz      = training_sz
        self.num_samples      = num_samples          # __len__
        self.dict_output      = dict_output           # if True, return dict instead of tuple
        self.same_pair        = same_pair            # if True, use the same image for both inputs
        # augment parameter ranges
        self.lower_sz         = param_ranges["lower_sz"]
        self.upper_sz         = param_ranges["upper_sz"]
        self.warp_pad         = param_ranges["warp_pad"]
        self.min_scale        = param_ranges["min_scale"]
        self.max_scale        = param_ranges["max_scale"]
        self.angle_range      = param_ranges["angle_range"]
        self.projective_range = param_ranges["projective_range"]
        self.translation_range= param_ranges["translation_range"]
        self.training_sz_pad  = round(training_sz + training_sz * 2 * self.warp_pad)

        # misc
        self.transform = transform or transforms.ToTensor()

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
        template = images[pair-1].copy()

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
        template_image = self.transform(template_crop)    # ditto

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
        ).view(8, 1)

        # ----------------- apply inverse warp to image ----------------- #
        #  → we want 'warped_image' such that template == warp(warped_image, p_gt)
        img_tensor_4d = img_tensor.unsqueeze(0)         # [1,3,H,W]
        H             = param_to_H(p_gt.unsqueeze(0))               # [1,3,3]
        H_inv         = torch.linalg.inv(H)
        img_w, _      = warp_hmg(img_tensor_4d, H_to_param(H_inv))
        warped_image  = img_w.squeeze(0)                                # back to [3,H,W]

        # ------------- remove padding to reach final size -------------- #
        pad_side = round(self.training_sz * self.warp_pad)
        warped_image  = warped_image[:, pad_side : pad_side + self.training_sz,
                                        pad_side : pad_side + self.training_sz]
        template_image = template_image[:, pad_side : pad_side + self.training_sz,
                                              pad_side : pad_side + self.training_sz]
        if self.dict_output:
            sample = {
                "img1":   template_image,
                "img2":   warped_image,
                "H_1_2":  H,
            }
            return sample
        return warped_image, template_image, p_gt
    
    
    
def default_parameter_ranges(training_sz: int):
    """
    Returns default parameter ranges for homography warping.
    """
    assert training_sz > 0, "Training size must be positive."
    parameter_ranges ={
        "lower_sz": training_sz,
        "upper_sz": training_sz,
        "warp_pad": 0.4,
        "min_scale": 1.0,
        "max_scale": 1.0,
        "angle_range": 3,
        "projective_range": 0,
        "translation_range": 5,
    }
    return parameter_ranges


def parameter_ranges(lower_sz, upper_sz, warp_pad, min_scale, max_scale, angle_range, projective_range, translation_range):
    """
    Returns parameter ranges for homography warping.
    """
    assert lower_sz > 0 and upper_sz > 0, "Sizes must be positive."
    assert min_scale > 0 and max_scale >= min_scale, "Scale must be positive."
    assert angle_range >= 0, "Angle range must be non-negative."
    assert projective_range >= 0, "Projective range must be non-negative."
    assert translation_range >= 0, "Translation range must be non-negative."
    
    return {
        "lower_sz": lower_sz,
        "upper_sz": upper_sz,
        "warp_pad": warp_pad,
        "min_scale": min_scale,
        "max_scale": max_scale,
        "angle_range": angle_range,
        "projective_range": projective_range,
        "translation_range": translation_range,
    }