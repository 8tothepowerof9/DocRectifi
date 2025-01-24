import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from illum.dataset import RealDAE
from illum.model import GCNet
from illum.trainer import GCTrainer


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


def vis_preds(model, dataloader):
    # Visualize multiple instances
    model.eval()

    with torch.no_grad():
        for inputs, gts in dataloader:
            # num_images = len(inputs)
            num_images = 2
            fig, axes = plt.subplots(num_images, 4, figsize=(15, 5 * num_images))

            # Ensure axes is a 2D array even if num_images == 1
            if num_images == 1:
                axes = axes.reshape(1, -1)

            for i in range(num_images):
                pred_shadowmap = model(inputs[i].unsqueeze(0).to("cuda"))
                i_gc = torch.clamp(
                    inputs[i].unsqueeze(0).to("cuda") / pred_shadowmap, 0, 1
                )

                # Convert back to HWC format for visualization
                inp_img = inputs[i].permute(1, 2, 0).numpy()
                gt_img = gts[i].permute(1, 2, 0).numpy()
                pred_shadowmap = pred_shadowmap.squeeze().permute(1, 2, 0).cpu().numpy()
                i_gc = i_gc.squeeze().permute(1, 2, 0).cpu().numpy()

                # Input Image
                axes[i, 0].imshow(inp_img)
                axes[i, 0].set_title("Input Image")
                axes[i, 0].axis("off")

                # Ground Truth Image
                axes[i, 1].imshow(gt_img)
                axes[i, 1].set_title("Ground Truth Image")
                axes[i, 1].axis("off")

                # Predicted Shadowmap
                axes[i, 2].imshow(pred_shadowmap)
                axes[i, 2].set_title("Predicted Shadowmap")
                axes[i, 2].axis("off")

                # Predicted Illumination
                axes[i, 3].imshow(i_gc)
                axes[i, 3].set_title("Predicted Illumination")
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
    model = GCNet(config).to("cuda")

    trainer = GCTrainer(
        model=model,
        config=config,
    )

    # print(model)

    trainer.fit(train_loader, val_loader)

    # vis_preds(model, train_loader)
