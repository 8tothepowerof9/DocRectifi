import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from illum.dataset import RealDAE
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


# TODO: Modify later
def vis_preds(dr, gc, dataloader):
    # Visualize multiple instances
    dr.eval()
    gc.eval()

    with torch.no_grad():
        for inputs, gts in dataloader:
            # num_images = len(inputs)
            num_images = 2
            _, axes = plt.subplots(num_images, 4, figsize=(15, 5 * num_images))

            # Ensure axes is a 2D array even if num_images == 1
            if num_images == 1:
                axes = axes.reshape(1, -1)

            in_img, gt_img = inputs.to("cuda"), gts.to("cuda")
            shadow_map = torch.clamp(in_img / (gt_img + 1e-6), 0, 1)
            pred_shadow_map = gc(in_img)

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

    train_ds = RealDAE(split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    val_ds = RealDAE(split="val")
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    # Get model
    dr = DRNet(config).to("cuda")
    gc = GCNet(config).to("cuda")

    trainer = GCDRTrainer(
        model=dr,
        config=config,
    )

    trainer.fit(train_loader, val_loader)

    vis_preds(dr, gc, train_loader)
