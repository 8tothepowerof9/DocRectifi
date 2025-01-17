import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
    ds_dir : str
        The root directory containing the dataset.
    split : str
        The dataset split to load ('train' or 'val').
    transform : callable, optional
        An optional transformation function to apply to both input and ground truth images.
    """

    def __init__(self, ds_dir, split, config, transform=None):
        in_paths = []
        gt_paths = []

        if split not in ["train", "val"]:
            raise ValueError("split must be either 'train' or 'val'")

        for root, _, filenames in os.walk(ds_dir):
            for filename in filenames:
                if re.match(r".*in.jpg", filename):
                    in_paths.append(os.path.join(root, filename))
                elif re.match(r".*gt.jpg", filename):
                    gt_paths.append(os.path.join(root, filename))

        self.config = config
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
                A.Resize(
                    self.config["data"]["input_h"], self.config["data"]["input_w"]
                ),
            ]
        )

    def __get_val_trans__(self):
        return A.Compose(
            [A.Resize(self.config["data"]["input_h"], self.config["data"]["input_w"])]
        )

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


if __name__ == "__main__":
    config = {
        "data": {
            "input_h": 512,
            "input_w": 512,
        }
    }

    ds_dir = "./data/RealDAE"
    split = "train"

    dataset = RealDAE(ds_dir=ds_dir, split=split, config=config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    print(len(dataloader))

    def visualize_samples(batch):
        inputs, gts = batch

        fig, axes = plt.subplots(len(inputs), 2, figsize=(10, 10))

        for i in range(len(inputs)):
            # Convert back to HWC format for visualization
            inp_img = inputs[i].permute(1, 2, 0).numpy()
            gt_img = gts[i].permute(1, 2, 0).numpy()

            axes[i, 0].imshow(inp_img)
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(gt_img)
            axes[i, 1].set_title("Ground Truth Image")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()

    def plot_shadow_map(batch):
        inputs, gts = batch

        fig, axes = plt.subplots(len(inputs), 4, figsize=(10, 10))

        for i in range(len(inputs)):
            # Compute shadowmap by dividing input by ground truth
            epsilon = 1e-8
            shadow_map = inputs[i] / (gts[i] + epsilon)

            # Concat input and shadow map
            i_gc = torch.cat([inputs[i], shadow_map], dim=0)

            # Convert back to HWC format for visualization
            i_gc = i_gc.permute(1, 2, 0).numpy()
            shadow_map = shadow_map.permute(1, 2, 0).numpy()
            inp_img = inputs[i].permute(1, 2, 0).numpy()
            gt_img = gts[i].permute(1, 2, 0).numpy()

            axes[i, 0].imshow(inp_img)
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(gt_img)
            axes[i, 1].set_title("Ground Truth Image")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(shadow_map)
            axes[i, 2].set_title("Shadow Map")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(i_gc)
            axes[i, 3].set_title("Input + Shadow Map")
            axes[i, 3].axis("off")

        plt.tight_layout()
        plt.show()

    # Get a batch of samples and visualize
    for batch in dataloader:
        # print(batch[0])
        # print(batch[1])
        # visualize_samples(batch)
        plot_shadow_map(batch)
        break  # Only visualize one batch
