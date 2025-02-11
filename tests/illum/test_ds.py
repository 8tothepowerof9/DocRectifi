import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from illum.dataset import RealDAE, FullResBatchSampler
from illum.utils import pad_to_stride

if __name__ == "__main__":
    split = "train"

    dataset = RealDAE(split=split)
    sampler = FullResBatchSampler(1, dataset.imgs_size_idx, shuffle=True)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=4)

    def visualize_samples(batch):
        inputs, gts = batch

        _, axes = plt.subplots(len(inputs), 2, figsize=(10, 10))

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

        _, axes = plt.subplots(len(inputs), 4, figsize=(10, 10))

        for i in range(len(inputs)):
            # Compute shadowmap by dividing input by ground truth
            shadow_map = inputs[i] / gts[i]  # May show clamp warnings from matplotlib

            i_gc = torch.clamp(inputs[i] / shadow_map, 0, 1)

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
            axes[i, 3].set_title("Enhanced map")
            axes[i, 3].axis("off")

        plt.tight_layout()
        plt.show()

    # Get a batch of samples and visualize
    # for idx, batch in enumerate(dataloader):
    #     print(batch[0].dtype)
    #     print(batch[1].dtype)
    #     # visualize_samples(batch)
    #     # plot_shadow_map(batch)
    #     # break  # Only visualize one batch
    #     continue

    for idx, (in_img, gt_img) in enumerate(dataloader):
        # Print original shape
        print("Original shape: ")
        print(in_img.shape, gt_img.shape)
        _, _, h, w = in_img.shape

        print("Shape after resizing short side to 512 while keeping the aspect ratio: ")
        short_side = min(h, w)
        if short_side < 512:
            scale = 512 / short_side
            h = int(h * scale)
            w = int(w * scale)
            in_img = F.interpolate(in_img, (h, w), mode="bilinear", align_corners=False)
            gt_img = F.interpolate(gt_img, (h, w), mode="bilinear", align_corners=False)
        print(in_img.shape, gt_img.shape)

        print("Shape after ensuring divisible by 32: ")
        in_img, padding_h, padding_w = pad_to_stride(in_img, stride=32)
        gt_img, _, _ = pad_to_stride(gt_img, stride=32)

        print(in_img.shape, gt_img.shape)
        print()
