import os
import re
import copy
import random
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import albumentations as A
import cv2
from sklearn.model_selection import train_test_split
from .config import *


class RealDAE(Dataset):
    """
    The dataset expects the directory to be organized such that files corresponding to the
    'train' or 'val' split are stored in appropriate subdirectories, with images following
    a naming convention that ends with 'in.jpg' for input images and 'gt.jpg' for ground truth images.
    Some transformations can't be used or else it will through an error during training. For example, RandomRotate90.

    Attributes:
    -----------
    in_paths : list of str
        Sorted list of file paths to input images.
    gt_paths : list of str
        Sorted list of file paths to ground truth images.

    Parameters:
    -----------
    split : str
        The dataset split to load ('train' or 'val').
    """

    def __init__(self, split):
        in_paths = []
        gt_paths = []

        if split not in ["train", "val"]:
            raise ValueError("split must be either 'train' or 'val'")

        for root, _, filenames in os.walk(DATASET_PATH):
            for filename in filenames:
                if re.match(r".*in.jpg", filename):
                    in_paths.append(os.path.join(root, filename))
                elif re.match(r".*gt.jpg", filename):
                    gt_paths.append(os.path.join(root, filename))

        self.in_paths = sorted(in_paths)
        self.gt_paths = sorted(gt_paths)

        # Split the dataset into train and val
        train_in_paths, val_in_paths, train_gt_paths, val_gt_paths = train_test_split(
            self.in_paths, self.gt_paths, test_size=0.2, random_state=42
        )

        if split == "train":
            self.in_paths = train_in_paths
            self.gt_paths = train_gt_paths
        elif split == "val":
            self.in_paths = val_in_paths
            self.gt_paths = val_gt_paths

        # Organize images by size
        imgs_size_idx = {}
        for idx, path in enumerate(self.in_paths):
            # Get image w and h
            w, h = cv2.imread(path).shape[:2]
            size = (w, h)
            if size not in imgs_size_idx:
                imgs_size_idx[size] = [idx]
            else:
                imgs_size_idx[size].append(idx)

        if split == "train":
            transform = self.__get_train_trans__()
        elif split == "val":
            transform = self.__get_val_trans__()

        self.imgs_size_idx = imgs_size_idx
        self.transform = transform

    def __get_train_trans__(self):
        transforms = [
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ]

        return A.Compose(transforms)

    def __get_val_trans__(self):
        transforms = []

        return A.Compose(transforms)

    def __len__(self):
        return len(self.in_paths)

    def __getitem__(self, idx):
        in_path = self.in_paths[idx]
        gt_path = self.gt_paths[idx]

        in_img = cv2.imread(in_path)
        gt_img = cv2.imread(gt_path)

        if self.transform:
            transformed = self.transform(image=in_img, mask=gt_img)
            in_img = transformed["image"]
            gt_img = transformed["mask"]

        # Reduce mem
        in_img = in_img.astype("float16") / 255.0
        in_img = in_img.transpose(2, 0, 1)
        gt_img = gt_img.astype("float16") / 255.0
        gt_img = gt_img.transpose(2, 0, 1)

        return in_img, gt_img


class FullResBatchSampler(Sampler):
    """
    Custom batch sampler that samples batches of images at full resolution.
    This sampler shuffles indices and drops last batch.

    """

    def __init__(self, batch_size, imgs_size_idx, shuffle=True):
        self.batch_size = batch_size
        self.imgs_size_idx = imgs_size_idx
        self.shuffle = shuffle

    def __iter__(self):
        imgsz_idxs = copy.deepcopy(self.imgs_size_idx)

        batches = []
        for size in imgsz_idxs:
            while len(imgsz_idxs[size]) >= self.batch_size:  # drop last
                batch_idxs = []
                for _ in range(self.batch_size):
                    idx = imgsz_idxs[size].pop()
                    batch_idxs.append(idx)
                batches.append(batch_idxs)
        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        batches_per_resolution = [
            len(idxs) // self.batch_size for idxs in self.imgs_size_idx.values()
        ]
        return int(sum(batches_per_resolution))
