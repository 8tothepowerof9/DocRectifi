import sys
import torch
from torch.utils.data import DataLoader
from models.illumination import *
from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)
import matplotlib.pyplot as plt
from scripts.illumination.dataset import RealDAE
from scripts.illumination.utils import read_cfg
import pandas as pd


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = self.__read_model__()
        self.log = self.__read_log__()
        self.dataloader = DataLoader(
            RealDAE(
                ds_dir="./data/RealDAE", split=cfg["eval"]["eval_loader"], config=cfg
            )
        )

    def __read_log__(self):
        path = f"results/logs/illumination/{self.cfg['model']['name']}.csv"
        return pd.read_csv(path)

    def __read_model__(self):
        if self.cfg["model"]["type"] == "baseline":
            model = Baseline(self.cfg)
        elif self.cfg["model"]["type"] == "unext":
            model = UNext(self.cfg)
        else:
            raise ValueError("Model type not supported")

        # load weights
        path = f"models/illumination/checkpoints/{self.cfg['model']['name']}.pt"
        model.load_state_dict(torch.load(path, weights_only=True))
        model.to("cuda")
        return model

    def vis_preds(self):
        # Visualize multiple instances
        self.model.eval()

        with torch.no_grad():
            for inputs, gts in self.dataloader:
                num_images = len(inputs)
                fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

                # Ensure axes is a 2D array even if num_images == 1
                if num_images == 1:
                    axes = axes.reshape(1, -1)

                for i in range(num_images):
                    # Convert back to HWC format for visualization
                    inp_img = inputs[i].permute(1, 2, 0).numpy()
                    gt_img = gts[i].permute(1, 2, 0).numpy()

                    # Input Image
                    axes[i, 0].imshow(inp_img)
                    axes[i, 0].set_title("Input Image")
                    axes[i, 0].axis("off")

                    # Ground Truth Image
                    axes[i, 1].imshow(gt_img)
                    axes[i, 1].set_title("Ground Truth Image")
                    axes[i, 1].axis("off")

                    # Predicted Image
                    pred_img = self.model(inputs[i].unsqueeze(0).to("cuda"))
                    axes[i, 2].imshow(pred_img[0].permute(1, 2, 0).cpu().numpy())
                    axes[i, 2].set_title("Predicted Image")
                    axes[i, 2].axis("off")

                plt.tight_layout()
                plt.show()
                break  # Visualize only one batch

            # Save plot if specified
            if self.cfg["eval"]["save"]:
                fig.savefig(
                    f"results/illumination/vis_preds_{self.cfg['model']['name']}.png"
                )

    def plot_history(self):
        epochs = range(1, len(self.log["loss"]) + 1)  # Number of epochs

        # Create a single figure with 3 subplots
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        # Plot loss and validation loss
        ax[0].plot(epochs, self.log["loss"], label="Training Loss", marker="o")
        ax[0].plot(epochs, self.log["val_loss"], label="Validation Loss", marker="o")
        ax[0].set_title("Loss vs Epochs")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        ax[0].grid()

        # Plot PSNR and validation PSNR
        ax[1].plot(epochs, self.log["PSNR"], label="Training PSNR", marker="o")
        ax[1].plot(epochs, self.log["val_PSNR"], label="Validation PSNR", marker="o")
        ax[1].set_title("PSNR vs Epochs")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("PSNR (dB)")
        ax[1].legend()
        ax[1].grid()

        # Plot MS-SSIM and validation MS-SSIM
        ax[2].plot(epochs, self.log["MS_SSIM"], label="Training MS-SSIM", marker="o")
        ax[2].plot(
            epochs, self.log["val_MS_SSIM"], label="Validation MS-SSIM", marker="o"
        )
        ax[2].set_title("MS-SSIM vs Epochs")
        ax[2].set_xlabel("Epochs")
        ax[2].set_ylabel("MS-SSIM")
        ax[2].legend()
        ax[2].grid()

        # Adjust layout
        fig.tight_layout()

        if self.cfg["eval"]["save"]:
            fig.savefig(f"results/illumination/history_{self.cfg['model']['name']}.png")

        plt.show()

    def eval(self):
        pass


if __name__ == "__main__":
    cfg_file = sys.argv[1]
    config = read_cfg(cfg_file)

    evaluator = Evaluator(cfg=config)

    if config["eval"]["plot_history"]:
        evaluator.plot_history()

    if config["eval"]["vis_preds"]:
        evaluator.vis_preds()
