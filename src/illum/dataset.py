import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from .config import *


class RealDAE(Dataset):
    """
    A PyTorch Dataset class for loading paired image data from a directory structure.
    This dataset is designed to handle image pairs for tasks such as denoising, inpainting,
    or any application requiring input (in) and ground truth (gt) images.

    The dataset expects the directory to be organized such that files corresponding to the
    'train' or 'val' split are stored in appropriate subdirectories, with images following
    a naming convention that ends with 'in.jpg' for input images and 'gt.jpg' for ground truth images.

    Attributes:
    -----------
    in_paths : list of str
        Sorted list of file paths to input images.
    gt_paths : list of str
        Sorted list of file paths to ground truth images.
    transform : callable, optional
        A function/transform to apply to the images (both input and ground truth).
        Only works with albumentations transforms.

    Parameters:
    -----------
    split : str
        The dataset split to load ('train' or 'val').
    transform : callable, optional
        An optional transformation function to apply to both input and ground truth images.
    """

    def __init__(self, split, transform=None):
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

        if not transform:
            if split == "train":
                transform = self.__get_train_trans__()
            elif split == "val":
                transform = self.__get_val_trans__()

        self.transform = transform

    def __get_train_trans__(self):
        return A.Compose(
            [
                A.RandomRotate90(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Resize(IMG_H, IMG_W),
            ]
        )

    def __get_val_trans__(self):
        return A.Compose([A.Resize(IMG_H, IMG_W)])

    def __len__(self):
        return len(self.in_paths)

    def __getitem__(self, idx):
        in_path = self.in_paths[idx]
        gt_path = self.gt_paths[idx]

        in_img = cv2.imread(in_path)
        gt_img = cv2.imread(gt_path)

        if self.transform:
            augmented = self.transform(image=in_img, mask=gt_img)
            in_img = augmented["image"]
            gt_img = augmented["mask"]

        in_img = in_img.astype("float32") / 255.0
        in_img = in_img.transpose(2, 0, 1)
        gt_img = gt_img.astype("float32") / 255.0
        gt_img = gt_img.transpose(2, 0, 1)

        return in_img, gt_img
