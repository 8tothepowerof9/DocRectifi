import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from illum.dataset import RealDAE, FullResBatchSampler
from illum.model import DRNet, GCNet
from illum.trainer import GCDRTrainer


def read_cfg(file_path):
    try:
        with open(file_path, "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"File not found at {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Invalid JSON format in {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


def vis_preds(dr, gc, dataloader):
    # Visualize multiple instances
    dr.eval()
    gc.eval()

    with torch.no_grad():
        for padded, gc_input, _ in dataloader:
            num_images = len(padded)
            _, axes = plt.subplots(num_images, 4, figsize=(15, 5 * num_images))

            # Ensure axes is a 2D array even if num_images == 1
            if num_images == 1:
                axes = axes.reshape(1, -1)

            in_img, gt_img, _, _ = padded
            in_img, gt_img = in_img.to("cuda"), gt_img.to("cuda")  # Padded images
            in_img_down, _ = gc_input
            in_img_down = in_img_down.to("cuda")
            _, _, h, w = gt_img.shape

            pred_shadow_map = gc(in_img_down)

            # Upscale to original size
            pred_shadow_map = F.interpolate(pred_shadow_map, (h, w), mode="nearest")
            i_gc = torch.clamp(in_img / pred_shadow_map, 0, 1)
            dr_input = torch.cat((in_img, i_gc), dim=1)

            _, _, _, out1 = dr(dr_input)

            # Plot input, ground truth, i_gc, and final output

            for i in range(num_images):
                axes[i, 0].imshow(in_img[i].cpu().permute(1, 2, 0))
                axes[i, 0].set_title("Input")
                axes[i, 0].axis("off")

                axes[i, 1].imshow(gt_img[i].cpu().permute(1, 2, 0))
                axes[i, 1].set_title("Ground Truth")
                axes[i, 1].axis("off")

                axes[i, 2].imshow(i_gc[i].cpu().permute(1, 2, 0))
                axes[i, 2].set_title("I_GC")
                axes[i, 2].axis("off")

                axes[i, 3].imshow(out1[i].cpu().permute(1, 2, 0))
                axes[i, 3].set_title("Output")
                axes[i, 3].axis("off")

            plt.tight_layout()
            plt.show()
            break  # Visualize only one batch


if __name__ == "__main__":
    cfg_file = sys.argv[1]
    config = read_cfg(cfg_file)

    train_ds = RealDAE(split="train", min_mem_usage=True)
    val_ds = RealDAE(split="val", min_mem_usage=True)

    # Batch sampler
    # train_sampler = FullResBatchSampler(
    #     config["data"]["batch_size"], train_ds.imgs_size_idx, shuffle=True
    # )
    val_sampler = FullResBatchSampler(
        config["data"]["batch_size"], val_ds.imgs_size_idx, shuffle=True
    )

    # train_loader = DataLoader(
    #     train_ds,
    #     num_workers=config["data"]["num_workers"],
    #     batch_sampler=train_sampler,
    #     pin_memory=True,
    # )

    val_loader = DataLoader(
        val_ds,
        num_workers=config["data"]["num_workers"],
        batch_sampler=val_sampler,
        pin_memory=True,
    )
    # Get model
    dr = DRNet(config).to("cuda")

    trainer = GCDRTrainer(
        model=dr,
        config=config,
    )

    #trainer.fit(train_loader, val_loader)

    gc = GCNet(config).to("cuda")
    vis_preds(dr, gc, val_loader)
