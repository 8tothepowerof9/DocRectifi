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


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = self.__read_model__()

    def __read_model__(self):
        if self.cfg["model"]["type"] == "baseline":
            model = Baseline(self.cfg)
        elif self.cfg["model"]["type"] == "unet":
            model = UNet(self.cfg)
        else:
            raise ValueError("Model type not supported")

        # load weights
        path = f"models/illumination/{self.cfg['model']['name']}.pt"
        model.load_state_dict(torch.load(path, weights_only=True)).to(
            "cuda"
        )  # Use GPU only
        return model

    def vis_pred(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            in_img, gt_img = next(iter(dataloader))
            pred_img = self.model(in_img.to("cuda"))

            _, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(in_img[0].permute(1, 2, 0).cpu().numpy())
            ax[0].set_title("Input Image")

            ax[1].imshow(gt_img[0].permute(1, 2, 0).cpu().numpy())
            ax[1].set_title("Ground Truth")

            ax[2].imshow(pred_img[0].permute(1, 2, 0).cpu().numpy())
            ax[2].set_title("Predicted Image")

            plt.show()

        # Save plot if specified
        if self.cfg["eval"]["save"]:
            plt.savefig(f"results/illumination/{self.cfg['model']['name']}.png")

    def vis_preds(self, dataloader):
        # Visualize multiple instances
        self.model.eval()

        with torch.no_grad():
            for inputs, gts in dataloader:
                _, axes = plt.subplots(len(inputs), 3, figsize=(15, 10))

                for i in range(
                    len(inputs)
                ):  # Convert back to HWC format for visualization
                    inp_img = inputs[i].permute(1, 2, 0).numpy()
                    gt_img = gts[i].permute(1, 2, 0).numpy()

                    axes[i, 0].imshow(inp_img)
                    axes[i, 0].set_title("Input Image")
                    axes[i, 0].axis("off")

                    axes[i, 1].imshow(gt_img)
                    axes[i, 1].set_title("Ground Truth Image")
                    axes[i, 1].axis("off")

                    # Predict
                    pred_img = self.model(inputs[i].unsqueeze(0).to("cuda"))
                    axes[i, 2].imshow(pred_img[0].permute(1, 2, 0).cpu().numpy())
                    axes[i, 2].set_title("Predicted Image")
                    axes[i, 2].axis("off")

                plt.tight_layout()
                plt.show()
                break  # One batch only

        # Save plot if specified
        if self.cfg["eval"]["save"]:
            plt.savefig(f"results/illumination/{self.cfg['model']['name']}.png")


if __name__ == "__main__":
    cfg_file = sys.argv[1]
    config = read_cfg(cfg_file)

    dataloader = DataLoader(
        RealDAE(
            ds_dir="./data/RealDAE", split=config["eval"]["eval_loader"], config=config
        )
    )

    evaluator = Evaluator(cfg=config)
    evaluator.vis_preds(dataloader)
