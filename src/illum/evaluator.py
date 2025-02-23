import pandas as pd
import torch
import matplotlib.pyplot as plt
from .model import *
from .config import *


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.models = self.__read_model__()
        self.log = self.__read_log__()

    def __read_log__(self):
        path = f"{LOGS_PATH}/{self.cfg['model']['name']}.csv"
        return pd.read_csv(path)

    def __read_model__(self):
        if self.cfg["model"]["type"] not in MODEL_LIST.keys():
            raise ValueError(f"Model {self.cfg['model']['type']} not supported")

        models = []

        # If type is gcdr, load both gcnet and drnet

        # load weights
        if self.cfg["model"]["type"] != "gcdr":
            model = MODEL_LIST[self.cfg["model"]["type"]](self.cfg)
            path = f"{CHECKPOINTS_PATH}/{self.cfg['model']['name']}.pt"
            model.load_state_dict(torch.load(path, weights_only=True))
            model.to("cuda")
            models.append(model)
        else:
            dr = MODEL_LIST[self.cfg["model"]["type"]](self.cfg)
            dr_path = f"{CHECKPOINTS_PATH}/{self.cfg['model']['dr']['name']}.pt"
            dr.load_state_dict(torch.load(dr_path, weights_only=True))
            dr.to("cuda")

            # Do the same to gcnet
            gc = MODEL_LIST["gcnet"](self.cfg)
            gc_path = f"{CHECKPOINTS_PATH}/{self.cfg['model']['gc']['name']}.pt"
            gc.load_state_dict(torch.load(gc_path, weights_only=True))
            gc.to("cuda")

            models.append(gc)
            models.append(dr)

        return models

    def vis_preds(self, dataloader):
        # Visualize multiple instances

        for model in self.models:
            model.eval()

        with torch.no_grad():
            for padded, _, _ in dataloader:
                inputs, gts, _, _ = padded
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

                    # Pass the input through pipeline
                    pred_img = None
                    for model in self.models:
                        i = inputs[i].unsqueeze(0).to("cuda")
                        pred_img = model(i)

                    axes[i, 2].imshow(pred_img[0].permute(1, 2, 0).cpu().numpy())
                    axes[i, 2].set_title("Predicted Image")
                    axes[i, 2].axis("off")

                plt.tight_layout()
                plt.show()
                break  # Visualize only one batch

            # Save plot if specified
            if self.cfg["eval"]["save"]:
                fig.savefig(f"{REPORT_PATH}/vis_preds_{self.cfg['model']['name']}.png")

    def plot_history(self):
        epochs = range(1, len(self.log["loss"]) + 1)  # Number of epochs

        # Create a single figure with 3 subplots
        fig, ax = plt.subplots(1, 4, figsize=(18, 5))

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

        ax[3].plot(epochs, self.log["lr"], label="Learning Rate", marker="o")
        ax[3].set_title("Learning Rate vs Epochs")
        ax[3].set_xlabel("Epochs")
        ax[3].set_ylabel("Learning Rate")
        ax[3].legend()
        ax[3].grid()

        # Adjust layout
        fig.tight_layout()

        if self.cfg["eval"]["save"]:
            fig.savefig(f"{REPORT_PATH}/history_{self.cfg['model']['name']}.png")

        plt.show()
