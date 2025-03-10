import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    StructuralSimilarityIndexMeasure as SSIM,
)
import os
import pandas as pd
from .base import BaseTrainer
from ..utils import (
    EarlyStopping,
    EarlyStoppingMultiModel,
    seconds_to_minutes_str,
)
from ..config import *
from ..model import GCNet
from ..loss import TVLoss


# GCNet trainer is similar to the StandardTrainer, but it uses the shadow map as the ground truth
class GCTrainer(BaseTrainer):
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

    def load_checkpoint(self):
        # Find if checkpoint exists, if so, load and return True, else return False
        checkpoint_path = f"{CHECKPOINTS_PATH}/{self.config['model']['gc']['name']}.pt"
        log_path = f"{LOGS_PATH}/{self.config['model']['gc']['name']}.csv"

        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

            if os.path.exists(log_path):
                self.log = pd.read_csv(log_path).to_dict(orient="list")

            return True
        else:
            return False

    def _train_epoch(self, dataloader):
        start = time.time()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        total_loss = 0

        # Metrics
        psnr = self.metrics["PSNR"]
        ssim = self.metrics["SSIM"]
        psnr.reset()
        ssim.reset()

        self.model.train()

        for batch, (_, gc, _) in enumerate(dataloader):
            in_img_down, shadow_map = gc
            in_img_down, shadow_map = in_img_down.to("cuda"), shadow_map.to("cuda")
            _, _, h, w = shadow_map.shape  # Original size

            # Forward pass
            pred = self.model(in_img_down)
            pred = F.interpolate(pred, size=(h, w), mode="nearest")

            loss = self.loss_fn(pred, shadow_map)
            total_loss += loss.item()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            ssim(pred, shadow_map)
            psnr(pred, shadow_map)

            # Logging
            if batch % 100 == 0:
                loss_val, current = loss.item(), batch * len(in_img_down)
                print(f"loss: {loss_val:.6f}  [{current:>5d}/{size:>5d}]")

        avg_loss = total_loss / num_batches
        ssim_score = ssim.compute().item()
        psnr_score = psnr.compute().item()

        # Save metrics
        lr = self.optimizer.param_groups[0]["lr"]
        self.log["loss"].append(avg_loss)
        self.log["PSNR"].append(psnr_score)
        self.log["SSIM"].append(ssim_score)
        self.log["lr"].append(lr)

        end = time.time()

        print(
            f"Train Summary [{end-start:.3f}s]: \n Avg Loss: {avg_loss:.4f} | SSIM: {ssim_score:.4f} | PSNR: {psnr_score:.4f} | lr: {lr}"
        )

        return end - start

    def _eval_epoch(self, dataloader):
        start = time.time()
        num_batches = len(dataloader)
        total_loss = 0

        # Metrics
        psnr = self.metrics["PSNR"]
        ssim = self.metrics["SSIM"]
        psnr.reset()
        ssim.reset()

        self.model.eval()

        with torch.no_grad():
            for _, (_, gc, _) in enumerate(dataloader):
                in_img_down, shadow_map = gc
                in_img_down, shadow_map = in_img_down.to("cuda"), shadow_map.to("cuda")
                _, _, h, w = shadow_map.shape  # Original size

                pred = self.model(in_img_down)
                pred = F.interpolate(pred, size=(h, w), mode="nearest")

                loss = self.loss_fn(pred, shadow_map)

                total_loss += loss.item()

                # Update metrics
                ssim(pred, shadow_map)
                psnr(pred, shadow_map)

        avg_loss = total_loss / num_batches
        ssim_score = ssim.compute().item()
        psnr_score = psnr.compute().item()

        # Save to log
        lr = self.optimizer.param_groups[0]["lr"]
        self.log["val_loss"].append(avg_loss)
        self.log["val_PSNR"].append(psnr_score)
        self.log["val_SSIM"].append(ssim_score)

        end = time.time()

        print(
            f"Validation Summary [{end-start:.3f}s]: \n Avg Loss: {avg_loss:.4f} | SSIM: {ssim_score:.4f} | PSNR: {psnr_score:.4f} | lr: {lr}"
        )

        return end - start

    def fit(self, train_loader, val_loader):
        if self.checkpoint_exists:
            print("----> Checkpoint found and loaded! <-----")

        print("-----Start Training!-----")
        early_stopper = EarlyStopping(patience=5, min_delta=0.01)

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
                f"{CHECKPOINTS_PATH}/{self.model.name}.pt",
            )
            print(f"----Best model from {early_stopper.best_model_epoch} saved!----")

            pd.DataFrame(self.log).to_csv(
                f"{LOGS_PATH}/{self.model.name}.csv", index=False
            )
            print("Logs saved!")

        print("-----Done Training!-----")


class GCDRTrainer(BaseTrainer):
    def __init__(self, model, config):
        # The model stored here is DRNet
        super().__init__(model, config)

        # Different lambda values for the loss function
        self.lambda_1 = 0.1
        self.lambda_2 = 0.002

        # l8 uses L1 loss only
        # l2 and l4 has the same loss function: SSIM loss + L1 loss
        # l1 uses L1 loss + SSIM loss + Total Variation loss
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM().to("cuda")  # For now use the default values
        self.tv_loss = TVLoss(p=1)

        # Attempt to load GCNet
        self.load_gcnet()
        self.load_checkpoint()

        self.gc_optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.gcnet.parameters()),
            lr=config["train"]["lr"],
            betas=tuple(config["train"]["betas"]),
        )
        self.dr_optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.gcnet.parameters()),
            lr=config["train"]["lr"],
            betas=tuple(config["train"]["betas"]),
        )

        self.epochs = self.config["train"]["epochs"]
        self.save = self.config["train"]["save"]

        if config["train"]["scheduler"]["type"] == "StepLR":
            self.gc_scheduler = optim.lr_scheduler.LinearLR(
                self.gc_optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.epochs,
            )
            self.dr_scheduler = optim.lr_scheduler.LinearLR(
                self.dr_optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.epochs,
            )
        elif config["train"]["scheduler"]["type"] == "LinearLR":
            self.gc_scheduler = optim.lr_scheduler.LinearLR(
                self.gc_optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.epochs,
            )
            self.dr_scheduler = optim.lr_scheduler.LinearLR(
                self.dr_optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.epochs,
            )
        else:
            raise ValueError("Scheduler type not recognized")

        self.min_lr = config["train"]["scheduler"]["min_lr"]

    def load_checkpoint(self):
        # Find if dr checkpoint exists, if so, load and return True, else return False
        checkpoint_path = f"{CHECKPOINTS_PATH}/{self.config['model']['dr']['name']}.pt"
        log_path = f"{LOGS_PATH}/{self.config['model']['dr']['name']}.csv"

        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

            if os.path.exists(log_path):
                self.log = pd.read_csv(log_path).to_dict(orient="list")

            self.dr_checkpoint_exists = True
        else:
            self.dr_checkpoint_exists = False

    def load_gcnet(self):
        self.gcnet = GCNet(self.config).to("cuda")
        checkpoint_path = f"{CHECKPOINTS_PATH}/{self.config['model']['gc']['name']}.pt"

        # Check if path exists
        if not os.path.exists(checkpoint_path):
            print("A GCNet checkpoint is needed to train GCDRNet.")
            raise FileNotFoundError(f"GCNet checkpoint not found at {checkpoint_path}")
        else:
            self.gcnet.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    def _train_epoch(self, dataloader):
        start = time.time()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        dr_total_loss = 0

        # Metrics
        dr_psnr = self.metrics["PSNR"]
        dr_ssim = self.metrics["SSIM"]
        dr_psnr.reset()
        dr_ssim.reset()

        self.model.train()
        self.gcnet.train()

        for batch, (padded, gc, dr) in enumerate(dataloader):
            # Unpack the data
            in_img, gt_img, _, _ = padded
            in_img, gt_img = in_img.to("cuda"), gt_img.to("cuda")

            # Input to gcnet
            in_img_down, shadow_map = gc
            in_img_down, shadow_map = in_img_down.to("cuda"), shadow_map.to("cuda")

            # Input to drnet
            gt8, gt4, gt2 = dr
            gt8, gt4, gt2 = gt8.to("cuda"), gt4.to("cuda"), gt2.to("cuda")

            _, _, h, w = shadow_map.shape  # Original size

            ## Train GCNet
            pred_shadow_map = self.gcnet(in_img_down)
            pred_shadow_map = F.interpolate(
                pred_shadow_map, size=(h, w), mode="nearest"
            )
            gc_loss = self.l1_loss(pred_shadow_map, shadow_map)

            # Gradient of the DR-Net will not propagate back to the GC-Net
            pred_shadow_map = pred_shadow_map.detach()
            i_gc = torch.clamp(in_img / pred_shadow_map, 0, 1)
            dr_input = torch.cat((in_img, i_gc), dim=1)

            ## Train DRNet
            out8, out4, out2, out1 = self.model(dr_input)

            ## Multiscale losses
            dr_loss = (
                self.l1_loss(out8, gt8)  # l8
                + self.l1_loss(out4, gt4)  # l4
                + self.lambda_1 * (1 - self.ssim_loss(out4, gt4))
                + self.l1_loss(out2, gt2)  # l2
                + self.lambda_1 * (1 - self.ssim_loss(out2, gt2))
                + self.l1_loss(out1, gt_img)  # l1
                + self.lambda_1 * (1 - self.ssim_loss(out1, gt_img))
                + self.lambda_2 * self.tv_loss(out1)
            )

            ## Backpropagation
            gc_loss.backward()
            dr_loss.backward()

            # Step the optimizers
            self.gc_optimizer.step()
            self.dr_optimizer.step()

            # Zero gradients for the combined optimizer
            self.gc_optimizer.zero_grad()
            self.dr_optimizer.zero_grad()

            ## Track
            with torch.no_grad():
                dr_total_loss += dr_loss.detach().item()
                dr_ssim(out1, gt_img)
                dr_psnr(out1, gt_img)

                # Logging
                if batch % 100 == 0:
                    dr_loss_val = dr_loss.detach().item()
                    current = batch * len(in_img)
                    print(f"loss: {dr_loss_val:.4f} [{current:>5d}/{size:>5d}]")

        lr = self.dr_optimizer.param_groups[0]["lr"]
        dr_avg_loss = dr_total_loss / num_batches
        dr_ssim_score = dr_ssim.compute().item()
        dr_psnr_score = dr_psnr.compute().item()

        # Save to log
        self.log["loss"].append(dr_avg_loss)
        self.log["PSNR"].append(dr_psnr_score)
        self.log["SSIM"].append(dr_ssim_score)
        self.log["lr"].append(lr)

        end = time.time()
        print(
            f"Train Summary [{end-start:.3f}s]: \n Avg Loss: {dr_avg_loss:.4f} | SSIM: {dr_ssim_score:.4f} | PSNR: {dr_psnr_score:.4f} | lr: {lr}"
        )

        return end - start

    def _eval_epoch(self, dataloader):
        start = time.time()
        num_batches = len(dataloader)
        dr_total_loss = 0

        # Metrics
        dr_psnr = self.metrics["PSNR"]
        dr_ssim = self.metrics["SSIM"]
        dr_psnr.reset()
        dr_ssim.reset()

        self.model.eval()
        self.gcnet.eval()

        with torch.no_grad():
            for padded, gc, dr in dataloader:
                # Unpack the data
                in_img, gt_img, _, _ = padded
                in_img, gt_img = in_img.to("cuda"), gt_img.to("cuda")

                # Input to gcnet
                in_img_down, shadow_map = gc
                in_img_down, shadow_map = in_img_down.to("cuda"), shadow_map.to("cuda")

                # Input to drnet
                gt8, gt4, gt2 = dr
                gt8, gt4, gt2 = gt8.to("cuda"), gt4.to("cuda"), gt2.to("cuda")

                _, _, h, w = shadow_map.shape  # Original size

                # GCNet
                pred_shadow_map = self.gcnet(in_img_down)

                ## Upscale the shadow map to the original size
                pred_shadow_map = F.interpolate(
                    pred_shadow_map, size=(h, w), mode="nearest"
                )
                i_gc = torch.clamp(in_img / pred_shadow_map, 0, 1)
                dr_input = torch.cat((in_img, i_gc), dim=1)

                # DRNet
                out8, out4, out2, out1 = self.model(dr_input)

                ## Multi-scale losses
                dr_loss = (
                    self.l1_loss(out8, gt8)
                    + self.l1_loss(out4, gt4)
                    + self.lambda_1 * (1 - self.ssim_loss(out4, gt4))
                    + self.l1_loss(out2, gt2)
                    + self.lambda_1 * (1 - self.ssim_loss(out2, gt2))
                    + self.l1_loss(out1, gt_img)
                    + self.lambda_1 * (1 - self.ssim_loss(out1, gt_img))
                    + self.lambda_2 * self.tv_loss(out1)
                )

                # Store total loss
                dr_total_loss += dr_loss.item()

                # Update metrics
                dr_ssim(out1, gt_img)
                dr_psnr(out1, gt_img)

        lr = self.dr_optimizer.param_groups[0]["lr"]
        dr_avg_loss = dr_total_loss / num_batches
        dr_ssim_score = dr_ssim.compute().item()
        dr_psnr_score = dr_psnr.compute().item()

        # Save to log
        self.log["val_loss"].append(dr_avg_loss)
        self.log["val_PSNR"].append(dr_psnr_score)
        self.log["val_SSIM"].append(dr_ssim_score)

        end = time.time()

        print(
            f"Eval Summary [{end-start:.3f}s]: \n Avg Loss: {dr_avg_loss:.4f} | SSIM: {dr_ssim_score:.4f} | PSNR: {dr_psnr_score:.4f} | lr: {lr}\n"
        )

        return end - start

    def fit(self, train_loader, val_loader):
        if self.dr_checkpoint_exists:
            print("----> DRNet checkpoint found and loaded! <-----")

        print("-----Start Training!-----")
        early_stopper = EarlyStoppingMultiModel(patience=5, min_delta=0.01)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            _ = self._train_epoch(train_loader)
            torch.cuda.empty_cache()
            _ = self._eval_epoch(val_loader)

            # Get lr
            lr = self.dr_optimizer.param_groups[0]["lr"]

            # lr cap
            if lr > self.min_lr:
                self.gc_scheduler.step()
                self.dr_scheduler.step()

            if early_stopper.early_stop(
                self.log["val_loss"][-1],
                self.model,
                self.gcnet,
                epoch + 1,
            ):
                print("Early Stopped!")
                break

        if self.save:
            torch.save(
                early_stopper.best_model_1_state,
                f"{CHECKPOINTS_PATH}/{self.model.name}.pt",
            )
            torch.save(
                early_stopper.best_model_2_state,
                f"{CHECKPOINTS_PATH}/{self.gcnet.name}.pt",
            )
            print(
                f"----Best GCNet and DRNet from {early_stopper.best_model_epoch} saved!----"
            )

            # Save logs
            pd.DataFrame(self.log).to_csv(
                f"{LOGS_PATH}/{self.model.name}.csv", index=False
            )
            print("Logs saved!")

        print("-----Done Training!-----")
