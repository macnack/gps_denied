from theseus.third_party.easyaug import GeoAugParam, RandomGeoAug, RandomPhotoAug
import logging
from PIL import Image
import numpy as np
import torch 
import cv2
from torch.utils.data import Dataset
import glob
import os


logger = logging.getLogger(__name__)


class HomographyDataset(Dataset):
    def __init__(self, img_dirs, imgH, imgW, photo_aug=True, train=True):
        self.imgH = imgH
        self.imgW = imgW
        self.img_paths = []
        for direc in img_dirs:
            self.img_paths.extend(glob.glob(direc + "/**/*.jpg", recursive=True))
        assert len(self.img_paths) > 0, "no images found"
        logger.info("Found %d total images in dataset" % len(self.img_paths))
        sc = 0.5
        self.rga = RandomGeoAug(
            rotate_param=GeoAugParam(min=-30 * sc, max=30 * sc),
            scale_param=GeoAugParam(min=(1.0 - 0.8 * sc), max=(1.0 + 1.2 * sc)),
            translate_x_param=GeoAugParam(min=-0.2 * sc, max=0.2 * sc),
            translate_y_param=GeoAugParam(min=-0.2 * sc, max=0.2 * sc),
            shear_x_param=GeoAugParam(min=-10 * sc, max=10 * sc),
            shear_y_param=GeoAugParam(min=-10 * sc, max=10 * sc),
            perspective_param=GeoAugParam(min=-0.1 * sc, max=0.1 * sc),
        )
        self.photo_aug = photo_aug
        if self.photo_aug:
            self.rpa = RandomPhotoAug()
            prob = 0.2  # Probability of augmentation applied.
            mag = 0.2  # Magnitude of augmentation [0: none, 1: max]
            self.rpa.set_all_probs(prob)
            self.rpa.set_all_mags(mag)

        # train test split
        self.img_paths.sort()
        max_images = 99999
        self.img_paths = self.img_paths[:max_images]
        split_ix = int(0.9 * len(self.img_paths))
        if train:
            self.img_paths = self.img_paths[:split_ix]
        else:
            self.img_paths = self.img_paths[split_ix:]
        self.train = train
        if self.train:
            logger.info("Using %d images for training" % len(self.img_paths))
        else:
            logger.info("Using %d images for testing" % len(self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img1 = np.asarray(Image.open(img_path).resize(size=(self.imgW, self.imgH)))
        # Convert file to rgb if it is grayscale.
        if img1.shape != (self.imgH, self.imgW, 3):
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img1 = torch.from_numpy(img1.astype(np.float32) / 255.0).permute(2, 0, 1)[None]

        # apply random geometric augmentations to create homography problem
        img2, H_1_2 = self.rga.forward(
            img1, return_transform=True, normalize_returned_transform=True
        )

        # apply random photometric augmentations
        if self.photo_aug:
            img1 = torch.clamp(img1, 0.0, 1.0)
            img2 = torch.clamp(img2, 0.0, 1.0)
            img1 = self.rpa.forward(img1)
            img2 = self.rpa.forward(img2)

        data = {"img1": img1[0], "img2": img2[0], "H_1_2": H_1_2[0]}

        return data
    
class HomographyAerialDataset(Dataset):
    def __init__(self, img_dirs, imgH, imgW, photo_aug=True, train=True):
        self.imgH = imgH
        self.imgW = imgW
        self.img_paths = []
        for direc in img_dirs:
            self.img_paths.extend(glob.glob(direc + "/**/*.jpg", recursive=True))
        assert len(self.img_paths) > 0, "no images found"
        logger.info("Found %d total images in dataset" % len(self.img_paths))
        sc = 0.5
        self.rga = RandomGeoAug(
            rotate_param=GeoAugParam(min=-30 * sc, max=30 * sc),
            scale_param=GeoAugParam(min=(1.0 - 0.8 * sc), max=(1.0 + 1.2 * sc)),
            translate_x_param=GeoAugParam(min=-0.2 * sc, max=0.2 * sc),
            translate_y_param=GeoAugParam(min=-0.2 * sc, max=0.2 * sc),
            shear_x_param=GeoAugParam(min=-10 * sc, max=10 * sc),
            shear_y_param=GeoAugParam(min=-10 * sc, max=10 * sc),
            perspective_param=GeoAugParam(min=-0.1 * sc, max=0.1 * sc),
        )
        self.photo_aug = photo_aug
        if self.photo_aug:
            self.rpa = RandomPhotoAug()
            prob = 0.2  # Probability of augmentation applied.
            mag = 0.2  # Magnitude of augmentation [0: none, 1: max]
            self.rpa.set_all_probs(prob)
            self.rpa.set_all_mags(mag)

        # train test split
        self.img_paths.sort()
        max_images = 99999
        self.img_paths = self.img_paths[:max_images]
        split_ix = int(0.9 * len(self.img_paths))
        if train:
            self.img_paths = self.img_paths[:split_ix]
        else:
            self.img_paths = self.img_paths[split_ix:]
        self.train = train
        if self.train:
            logger.info("Using %d images for training" % len(self.img_paths))
        else:
            logger.info("Using %d images for testing" % len(self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img1 = np.asarray(Image.open(img_path).resize(size=(self.imgW, self.imgH)))
        # Convert file to rgb if it is grayscale.
        if img1.shape != (self.imgH, self.imgW, 3):
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img1 = torch.from_numpy(img1.astype(np.float32) / 255.0).permute(2, 0, 1)[None]

        # apply random geometric augmentations to create homography problem
        img2, H_1_2 = self.rga.forward(
            img1, return_transform=True, normalize_returned_transform=True
        )

        # apply random photometric augmentations
        if self.photo_aug:
            img1 = torch.clamp(img1, 0.0, 1.0)
            img2 = torch.clamp(img2, 0.0, 1.0)
            img1 = self.rpa.forward(img1)
            img2 = self.rpa.forward(img2)

        data = {"img1": img1[0], "img2": img2[0], "H_1_2": H_1_2[0]}

        return data
    
    
# Download and extract data
def prepare_data():
    dataset_root = os.path.join(os.getcwd(), "data")
    chunks = [
        "revisitop1m.1",
        # "revisitop1m.2", # Uncomment for more data.
        # "revisitop1m.3",
        # "revisitop1m.4",
        # "revisitop1m.5",
    ]
    dataset_paths = []
    for chunk in chunks:
        dataset_path = os.path.join(dataset_root, chunk)
        dataset_paths.append(dataset_path)
        if not os.path.exists(dataset_path):
            logger.info("Downloading data")
            url_root = "http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/"
            tar = "%s.tar.gz" % chunk
            os.makedirs(dataset_path)
            cmd = "wget %s/%s -O %s/%s" % (url_root, tar, dataset_root, tar)
            logger.info("Running command: ", cmd)
            os.system(cmd)
            cmd = "tar -xf %s/%s -C %s" % (dataset_root, tar, dataset_path)
            logger.info("Running command: ", cmd)
            os.system(cmd)

    return dataset_paths