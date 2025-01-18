import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from illum.dataset import RealDAE

if __name__ == "__main__":
    split = "train"

    dataset = RealDAE(split=split)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    print(len(dataloader))

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

        fig, axes = plt.subplots(len(inputs), 4, figsize=(10, 10))

        for i in range(len(inputs)):
            # Compute shadowmap by dividing input by ground truth
            shadow_map = inputs[i] / gts[i]
            # Don't clamp shadow map

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
    for batch in dataloader:
        # print(batch[0])
        # print(batch[1])
        # visualize_samples(batch)
        plot_shadow_map(batch)
        break  # Only visualize one batch
