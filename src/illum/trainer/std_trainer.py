import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import pandas as pd
from .base import BaseTrainer
from ..utils import EarlyStopping, seconds_to_minutes_str
from ..config import *


class StandardTrainer(BaseTrainer):
    """
    The standard trainer is used to train models with simple training loops.
    For example, training a model with a single loss function and a single optimizer.
    For more complicated training loops, a custom Trainer class should be implmeneted by subclassing BaseTrainer.
    This class use the L1 loss function and Adam optimizer, and resize the input and target images to IMG_HxIMG_W.
    """

    def __init__(self, model, config):
        super().__init__(model, config)
        self.loss_fn = nn.L1Loss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config["train"]["lr"],
            betas=tuple(config["train"]["betas"]),
        )
        self.epochs = self.config["train"]["epochs"]
        self.save = self.config["train"]["save"]

        if config["train"]["scheduler"]["type"] == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config["train"]["scheduler"]["step_size"],
                gamma=config["train"]["scheduler"]["gamma"],
            )
        elif config["train"]["scheduler"]["type"] == "LinearLR":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.epochs,
            )
        else:
            raise ValueError("Scheduler type not recognized")

        self.min_lr = config["train"]["scheduler"]["min_lr"]
        self.checkpoint_exists = self.load_checkpoint()

    def _train_epoch(self, dataloader):
        start = time.time()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        total_loss = 0

        # Metrics
        psnr = self.metrics["PSNR"]
        ms_ssim = self.metrics["MS_SSIM"]
        psnr.reset()
        ms_ssim.reset()

        self.model.train()

        for batch, (in_img, gt_img) in enumerate(dataloader):
            in_img, gt_img = in_img.to("cuda"), gt_img.to("cuda")

            # Resize to IMG_HxIMG_W
            in_img = F.interpolate(in_img, (IMG_H, IMG_W), mode="bilinear")
            gt_img = F.interpolate(gt_img, (IMG_H, IMG_W), mode="bilinear")

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
            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(in_img)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        avg_loss = total_loss / num_batches
        ms_ssim_score = ms_ssim.compute().item()
        psnr_score = psnr.compute().item()

        # Save metrics
        lr = self.optimizer.param_groups[0]["lr"]
        self.log["loss"].append(avg_loss)
        self.log["PSNR"].append(psnr_score)
        self.log["MS_SSIM"].append(ms_ssim_score)
        self.log["lr"].append(lr)

        end = time.time()

        print(
            f"Train Summary [{end-start:.3f}s]: \n Avg Loss: {avg_loss:.4f} | MS-SSIM: {ms_ssim_score:.4f} | PSNR: {psnr_score:.4f} | lr: {lr}"
        )

        return end - start

    def _eval_epoch(self, dataloader):
        start = time.time()
        num_batches = len(dataloader)
        total_loss = 0

        # Metrics
        psnr = self.metrics["PSNR"]
        ms_ssim = self.metrics["MS_SSIM"]
        psnr.reset()
        ms_ssim.reset()

        self.model.eval()

        with torch.no_grad():
            for in_img, gt_img in dataloader:
                in_img, gt_img = in_img.to("cuda"), gt_img.to("cuda")

                # Resize to IMG_HxIMG_W
                in_img = F.interpolate(in_img, (IMG_H, IMG_W), mode="bilinear")
                gt_img = F.interpolate(gt_img, (IMG_H, IMG_W), mode="bilinear")

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
        lr = self.optimizer.param_groups[0]["lr"]
        self.log["val_loss"].append(avg_loss)
        self.log["val_PSNR"].append(psnr_score)
        self.log["val_MS_SSIM"].append(ms_ssim_score)

        end = time.time()

        print(
            f"Validation Summary [{end-start:.3f}s]: \n Avg Loss: {avg_loss:.4f} | MS-SSIM: {ms_ssim_score:.4f} | PSNR: {psnr_score:.4f} | lr: {lr}"
        )

        return end - start

    def fit(self, train_loader, val_loader):
        if self.checkpoint_exists:
            print("----> Checkpoint found and loaded! <-----")

        print("-----Start Training!-----")
        early_stopper = EarlyStopping(patience=3, min_delta=0.001)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            train_time = self._train_epoch(train_loader)
            val_time = self._eval_epoch(val_loader)

            if epoch > 0:
                print(
                    "\n[Approximate time remaining]: ",
                    seconds_to_minutes_str(
                        (train_time + val_time) * (self.epochs - epoch - 1)
                    ),
                    "\n",
                )

            # Get lr
            lr = self.optimizer.param_groups[0]["lr"]

            # lr cap
            if lr > self.min_lr:
                self.scheduler.step()

            if early_stopper.early_stop(
                self.log["val_loss"][-1], self.model, epoch + 1
            ):
                print("Early Stopped!")
                break

        if self.save:
            torch.save(
                early_stopper.best_model_state,
                "checkpoints/illum/" + self.model.name + ".pt",
            )
            print(f"----Best model from {early_stopper.best_model_epoch} saved!----")

            pd.DataFrame(self.log).to_csv(
                f"logs/illum/{self.model.name}.csv", index=False
            )
            print("Logs saved!")

        print("-----Done Training!-----")
