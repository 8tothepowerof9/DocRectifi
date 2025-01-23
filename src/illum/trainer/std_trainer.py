import time
import torch
from torch import nn, optim
import pandas as pd
from .base import BaseTrainer
from ..utils import EarlyStopping, seconds_to_minutes_str


class StandardTrainer(BaseTrainer):
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
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config["train"]["scheduler"]["step_size"],
            gamma=config["train"]["scheduler"]["gamma"],
        )
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

    def fit(self, train_loader, val_loader, min_lr=1e-6):
        if self.checkpoint_exists:
            print("----> Checkpoint found and loaded! <-----")

        print("-----Start Training!-----")
        early_stopper = EarlyStopping(patience=3, min_delta=0.001)

        for epoch in range(self.epochs):
            print(f"Epoch {len(self.log)+1}\n-------------------------------")
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
            if lr > min_lr:
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
