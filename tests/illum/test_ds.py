import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from illum.dataset import RealDAE, FullResBatchSampler
from illum.utils import pad_to_stride, remove_padding

if __name__ == "__main__":
    split = "val"

    dataset = RealDAE(split=split, min_mem_usage=True)
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

    for batch, (padded, gc, dr) in enumerate(dataloader):
        in_img, gt_img, padding_h, padding_w = padded
        in_img_down, shadow_map = gc
        gt8, gt4, gt2 = dr

        # Print shapes
        print("Input shape:", in_img.shape)
        print("Ground truth shape:", gt_img.shape)

        print("Input down shape:", in_img_down.shape)
        print("Shadow map shape:", shadow_map.shape)

        print("GT8 shape:", gt8.shape)
        print("GT4 shape:", gt4.shape)
        print("GT2 shape:", gt2.shape)

        # Plot in_img and gt_img
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(in_img[0].permute(1, 2, 0).numpy())
        plt.title("Padded Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(gt_img[0].permute(1, 2, 0).numpy())
        plt.title("Ground Truth Image")
        plt.axis("off")
        plt.show()

        break

    # for idx, (in_img, gt_img) in enumerate(dataloader):
    #     # Pad to stride 32
    #     new_in, padding_h, padding_w = pad_to_stride(in_img, stride=32)

    #     # Remove padding
    #     unpad_img = remove_padding(new_in, padding_h, padding_w)

    #     print("Original shape:", in_img.shape)
    #     print("Padded shape:", new_in.shape)
    #     print("Unpadded shape:", unpad_img.shape)

    #     # Plot the padded image and the original image side by side
    #     plt.figure(figsize=(10, 5))
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(new_in[0].permute(1, 2, 0).numpy())
    #     plt.title("Padded Image")
    #     plt.axis("off")
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(in_img[0].permute(1, 2, 0).numpy())
    #     plt.title("Original Image")
    #     plt.axis("off")
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(unpad_img[0].permute(1, 2, 0).numpy())
    #     plt.title("Unpadded Image")
    #     plt.axis("off")
    #     plt.show()

    #     break
