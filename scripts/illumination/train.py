# Different train, eval functions for GC-Net, and joint training of GC-Net and DR-Net
# Use the train function for GC-Net as the general training function for baseline

import time
import sys
import torch
from torch.utils.data import DataLoader
from scripts.illumination.dataset import RealDAE
from torch import nn, optim
from models.illumination import *
from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)
from scripts.illumination.utils import EarlyStopping, read_cfg
import pandas as pd
import matplotlib.pyplot as plt


class Trainer:
    def __init__(
        self,
        dataloaders,
        model,
        loss_fn,
        optimizer,
        device,
        scheduler=None,
        save=True,
        model_type="baseline",
    ):
        self.dataloaders = dataloaders
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.save = save
        self.metrics = {
            "PSNR": PSNR().to(device=device),
            "MS_SSIM": MS_SSIM().to(device=device),
        }  # For now use default parameters

        if model_type not in ["baseline", "gcdr"]:
            raise ValueError("Model type not supported")

        self.model_type = model_type
        self.log = {
            "loss": [],
            "val_loss": [],
            "PSNR": [],
            "MS_SSIM": [],
            "val_PSNR": [],
            "val_MS_SSIM": [],
        }

    def __std_train__(self):
        start = time.time()
        size = len(self.dataloaders["train"].dataset)
        num_batches = len(self.dataloaders["train"])
        total_loss = 0

        # Metrics
        psnr = self.metrics["PSNR"]
        ms_ssim = self.metrics["MS_SSIM"]
        psnr.reset()
        ms_ssim.reset()

        self.model.train()

        for batch, (in_img, gt_img) in enumerate(self.dataloaders["train"]):
            in_img, gt_img = in_img.to(self.device), gt_img.to(self.device)

            pred_gt = self.model(in_img)

            loss = self.loss_fn(pred_gt, gt_img)
            total_loss += loss.item()

            # Backprobagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            ms_ssim(pred_gt, gt_img)
            psnr(pred_gt, gt_img)

            # Logging
            if batch % 20 == 0:
                loss, current = loss.item(), batch * len(in_img)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if self.scheduler:
            self.scheduler.step()

        avg_loss = total_loss / num_batches
        ms_ssim_score = ms_ssim.compute().item()
        psnr_score = psnr.compute().item()

        # Save metrics
        self.log["loss"].append(avg_loss)
        self.log["PSNR"].append(psnr_score)
        self.log["MS_SSIM"].append(ms_ssim_score)

        end = time.time()

        print(
            f"Train Summary [{end-start:.3f}s]: \n Avg Loss: {avg_loss:.4f} | MS-SSIM: {ms_ssim_score:.4f} | PSNR: {psnr_score:.4f}"
        )

    def __std_eval__(self):
        start = time.time()
        num_batches = len(self.dataloaders["val"])
        total_loss = 0

        # Metrics
        psnr = self.metrics["PSNR"]
        ms_ssim = self.metrics["MS_SSIM"]
        psnr.reset()
        ms_ssim.reset()

        self.model.eval()

        with torch.no_grad():
            for in_img, gt_img in self.dataloaders["val"]:
                in_img, gt_img = in_img.to(self.device), gt_img.to(self.device)
                pred_gt = self.model(in_img)

                loss = self.loss_fn(pred_gt, gt_img)
                total_loss += loss.item()

                # Update metrics
                ms_ssim(pred_gt, gt_img)
                psnr(pred_gt, gt_img)

        avg_loss = total_loss / num_batches
        ms_ssim_score = ms_ssim.compute().item()
        psnr_score = psnr.compute().item()

        # Save to log
        self.log["val_loss"].append(avg_loss)
        self.log["val_PSNR"].append(psnr_score)
        self.log["val_MS_SSIM"].append(ms_ssim_score)

        end = time.time()

        print(
            f"Validation Summary [{end-start:.3f}s]: \n Avg Loss: {avg_loss:.4f} | MS-SSIM: {ms_ssim_score:.4f} | PSNR: {psnr_score:.4f}"
        )

    def __gcdr_train__(self):
        pass

    def __gcdr_eval__(self):
        pass

    def fit(self, epochs):
        early_stopper = EarlyStopping(patience=3, min_delta=0.001)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            if self.model_type == "baseline":
                self.__std_train__()
                self.__std_eval__()
            elif self.model_type == "gcdr":
                self.__gcdr_train__()
                self.__gcdr_eval__()

            if early_stopper.early_stop(self.log["val_loss"][-1], self.model):
                print("Early Stopped!")
                break

        if self.save:
            torch.save(
                early_stopper.best_model_state,
                "models/checkpoints/" + self.model.name + ".pt",
            )
            print("----Best model saved!----")

            pd.DataFrame(self.log).to_csv(
                f"results/logs/{self.model.name}.csv", index=False
            )
            print("Logs saved!")

        print("-----Done Training!-----")

        return self.log


if __name__ == "__main__":
    cfg_file = sys.argv[1]
    config = read_cfg(cfg_file)

    # Temporary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders = {
        "train": DataLoader(
            RealDAE(
                ds_dir="./data/RealDAE",
                split="train",
                config=config,
            ),
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        ),
        "val": DataLoader(
            RealDAE(
                ds_dir="./data/RealDAE",
                split="val",
                config=config,
            ),
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        ),
    }

    if config["model"]["type"] == "baseline":
        model = Baseline(cfg=config).to(device)
    elif config["model"]["type"] == "unet":
        model = UNet(cfg=config).to(device)
    else:
        raise ValueError("Model type not supported")

    trainer = Trainer(
        dataloaders=dataloaders,
        model=model,
        loss_fn=nn.L1Loss(),
        optimizer=optim.Adam(model.parameters(), lr=config["training"]["lr"]),
        device=device,
        save=False,
    )

    trainer.fit(config["training"]["epochs"])

    # Plot out a sample of the prediction
    model.eval()
    with torch.no_grad():
        in_img, gt_img = next(iter(dataloaders["val"]))
        pred_img = model(in_img.to(device))

        print(pred_img)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(in_img[0].permute(1, 2, 0).cpu().numpy())
        ax[0].set_title("Input Image")

        ax[1].imshow(gt_img[0].permute(1, 2, 0).cpu().numpy())
        ax[1].set_title("Ground Truth")

        ax[2].imshow(pred_img[0].permute(1, 2, 0).cpu().numpy())
        ax[2].set_title("Predicted Image")

        plt.show()
