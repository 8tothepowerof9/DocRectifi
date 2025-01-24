import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
    StructuralSimilarityIndexMeasure as SSIM,
    TotalVariation as TV,
)
import os
import pandas as pd
from .base import BaseTrainer
from ..utils import EarlyStopping, EarlyStoppingMultiModel, seconds_to_minutes_str
from ..config import CHECKPOINTS_PATH, LOGS_PATH
from ..model import GCNet


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
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config["train"]["scheduler"]["step_size"],
            gamma=config["train"]["scheduler"]["gamma"],
        )
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
        ms_ssim = self.metrics["MS_SSIM"]
        psnr.reset()
        ms_ssim.reset()

        self.model.train()

        for batch, (in_img, gt_img) in enumerate(dataloader):
            in_img, gt_img = in_img.to("cuda"), gt_img.to("cuda")

            # Calculate the shadowmap by dividing the in_img by gt_img
            # The shadow map will be clamped, and use in the loss
            shadow_map = torch.clamp(in_img / (gt_img + 1e-8), 0, 1)

            pred = self.model(in_img)

            loss = self.loss_fn(pred, shadow_map)
            total_loss += loss.item()

            # Backprobagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            ms_ssim(pred, shadow_map)
            psnr(pred, shadow_map)

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
                shadow_map = torch.clamp(in_img / (gt_img + 1e-6), 0, 1)

                pred = self.model(in_img)

                loss = self.loss_fn(pred, shadow_map)
                total_loss += loss.item()

                # Update metrics
                ms_ssim(pred, shadow_map)
                psnr(pred, shadow_map)

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
        self.tv_loss = TV(reduction="mean").to("cuda")  # Maybe use mean instead of sum?

        # Attempt to load GCNet
        self.load_gcnet()
        self.load_checkpoint()

        # TODO: Consider using 2 different optimizers
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.gcnet.parameters()),
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

        # Independent log and metrics for GCNet
        self.gc_log = {
            "loss": [],
            "val_loss": [],
            "PSNR": [],
            "MS_SSIM": [],
            "val_PSNR": [],
            "val_MS_SSIM": [],
            "lr": [],
        }

        self.gc_metrics = {
            "PSNR": PSNR().to(device="cuda"),
            "MS_SSIM": MS_SSIM().to(device="cuda"),
        }

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
        checkpoint_path = f"{CHECKPOINTS_PATH}/{self.config["model"]["gc"]["name"]}.pt"
        log_path = f"{LOGS_PATH}/{self.config['model']['gc']['name']}.csv"

        # Check if path exists
        if not os.path.exists(checkpoint_path):
            print("A GCNet checkpoint is needed to train GCDRNet.")
            raise FileNotFoundError(f"GCNet checkpoint not found at {checkpoint_path}")
        else:
            self.gcnet.load_state_dict(torch.load(checkpoint_path, weights_only=True))

            if os.path.exists(log_path):
                self.gc_log = pd.read_csv(log_path).to_dict(orient="list")

    def _train_epoch(self, dataloader):
        start = time.time()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        gc_total_loss = 0
        dr_total_loss = 0

        # Metrics
        dr_psnr = self.metrics["PSNR"]
        dr_ms_ssim = self.metrics["MS_SSIM"]
        dr_psnr.reset()
        dr_ms_ssim.reset()

        gc_psnr = self.gc_metrics["PSNR"]
        gc_ms_ssim = self.gc_metrics["MS_SSIM"]
        gc_psnr.reset()
        gc_ms_ssim.reset()

        self.model.train()
        self.gcnet.train()

        for batch, (in_img, gt_img) in enumerate(dataloader):
            in_img, gt_img = in_img.to("cuda"), gt_img.to("cuda")

            # Zero gradients for the combined optimizer
            self.optimizer.zero_grad()

            # Joint training
            ## Train GCNet
            shadow_map = torch.clamp(in_img / (gt_img + 1e-6), 0, 1)
            pred_shadow_map = self.gcnet(in_img)
            gc_loss = self.l1_loss(pred_shadow_map, shadow_map)
            gc_total_loss += gc_loss.item()

            ### Backpropagation for GCNet, retain graph for DRNet
            gc_loss.backward(retain_graph=True)

            ### Gradient of the DR-Net will not propagate back to the GC-Net
            pred_shadow_map = pred_shadow_map.detach()
            i_gc = torch.clamp(in_img / pred_shadow_map, 0, 1)
            dr_input = torch.cat((in_img, i_gc), dim=1)

            ## Train DRNet
            out8, out4, out2, out1 = self.model(dr_input)
            ### Resize gt to match the output sizes
            gt8, gt4, gt2, gt1 = (
                F.interpolate(
                    gt_img,
                    size=out8.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                ),
                F.interpolate(
                    gt_img,
                    size=out4.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                ),
                F.interpolate(
                    gt_img,
                    size=out2.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                ),
                gt_img,
            )

            ### Multi-scale losses
            l8 = self.l1_loss(out8, gt8)
            l4 = self.l1_loss(out4, gt4) + self.lambda_1 * (
                1 - self.ssim_loss(out4, gt4)
            )
            l2 = self.l1_loss(out2, gt2) + self.lambda_1 * (
                1 - self.ssim_loss(out2, gt2)
            )
            l1 = (
                self.l1_loss(out1, gt1)
                + self.lambda_1 * (1 - self.ssim_loss(out1, gt1))
                + self.lambda_2 * self.tv_loss(out1)
            )
            dr_loss = l8 + l4 + l2 + l1
            dr_total_loss += dr_loss.item()

            ### Backpropation for DRNet
            dr_loss.backward()
            self.optimizer.step()  # Optimizer step after both losses

            ### Update metrics
            gc_ms_ssim(pred_shadow_map, shadow_map)
            gc_psnr(pred_shadow_map, shadow_map)
            dr_ms_ssim(out1, gt1)
            dr_psnr(out1, gt1)

            # Logging
            if batch % 10 == 0:
                loss, gc_loss, dr_loss, current = (
                    gc_loss.item() + dr_loss.item(),
                    gc_loss.item(),
                    dr_loss.item(),
                    batch * len(in_img),
                )
                print(
                    f"loss: {loss:>4f} | gc_loss: {gc_loss:>4f} | dr_loss: {dr_loss:>4f} [{current:>5d}/{size:>5d}]"
                )

        lr = self.optimizer.param_groups[0]["lr"]
        gc_avg_loss = gc_total_loss / num_batches
        dr_avg_loss = dr_total_loss / num_batches
        gc_ms_ssim_score = gc_ms_ssim.compute().item()
        gc_psnr_score = gc_psnr.compute().item()
        dr_ms_ssim_score = dr_ms_ssim.compute().item()
        dr_psnr_score = dr_psnr.compute().item()
        avg_loss = gc_avg_loss + dr_avg_loss

        # Save to log
        self.log["loss"].append(dr_avg_loss)
        self.log["PSNR"].append(dr_psnr_score)
        self.log["MS_SSIM"].append(dr_ms_ssim_score)
        self.log["lr"].append(lr)

        self.gc_log["loss"].append(gc_avg_loss)
        self.gc_log["PSNR"].append(gc_psnr_score)
        self.gc_log["MS_SSIM"].append(gc_ms_ssim_score)
        self.gc_log["lr"].append(lr)

        end = time.time()

        print(
            f"Train Summary [{end-start:.3f}s]: \n Total Avg Loss: {avg_loss:.4f} | GC Avg Loss: {gc_avg_loss:.4f} | GC MS-SSIM: {gc_ms_ssim_score:.4f} | GC PSNR: {gc_psnr_score:.4f} | DR Avg Loss: {dr_avg_loss:.4f} | DR MS-SSIM: {dr_ms_ssim_score:.4f} | DR PSNR: {dr_psnr_score:.4f} | lr: {lr}"
        )

        return end - start

    def _eval_epoch(self, dataloader):
        start = time.time()
        num_batches = len(dataloader)
        gc_total_loss = 0
        dr_total_loss = 0

        # Metrics
        dr_psnr = self.metrics["PSNR"]
        dr_ms_ssim = self.metrics["MS_SSIM"]
        dr_psnr.reset()
        dr_ms_ssim.reset()

        gc_psnr = self.gc_metrics["PSNR"]
        gc_ms_ssim = self.gc_metrics["MS_SSIM"]
        gc_psnr.reset()
        gc_ms_ssim.reset()

        self.model.eval()
        self.gcnet.eval()

        with torch.no_grad():
            for in_img, gt_img in dataloader:
                in_img, gt_img = in_img.to("cuda"), gt_img.to("cuda")

                # GCNet
                shadow_map = torch.clamp(in_img / (gt_img + 1e-6), 0, 1)
                pred_shadow_map = self.gcnet(in_img)
                gc_loss = self.l1_loss(pred_shadow_map, shadow_map)
                gc_total_loss += gc_loss.item()

                i_gc = torch.clamp(in_img / pred_shadow_map, 0, 1)
                dr_input = torch.cat((in_img, i_gc), dim=1)

                # DRNet
                out8, out4, out2, out1 = self.model(dr_input)
                gt8, gt4, gt2, gt1 = (
                    F.interpolate(
                        gt_img,
                        size=out8.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                    F.interpolate(
                        gt_img,
                        size=out4.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                    F.interpolate(
                        gt_img,
                        size=out2.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                    gt_img,
                )

                ## Multi-scale losses
                l8 = self.l1_loss(out8, gt8)
                l4 = self.l1_loss(out4, gt4) + self.lambda_1 * (
                    1 - self.ssim_loss(out4, gt4)
                )
                l2 = self.l1_loss(out2, gt2) + self.lambda_1 * (
                    1 - self.ssim_loss(out2, gt2)
                )
                l1 = (
                    self.l1_loss(out1, gt1)
                    + self.lambda_1 * (1 - self.ssim_loss(out1, gt1))
                    + self.lambda_2 * self.tv_loss(out1)
                )

                dr_loss = l8 + l4 + l2 + l1
                dr_total_loss += dr_loss.item()

                # Update metrics
                gc_ms_ssim(pred_shadow_map, shadow_map)
                gc_psnr(pred_shadow_map, shadow_map)
                dr_ms_ssim(out1, gt1)
                dr_psnr(out1, gt1)

        lr = self.optimizer.param_groups[0]["lr"]
        gc_avg_loss = gc_total_loss / num_batches
        dr_avg_loss = dr_total_loss / num_batches
        gc_ms_ssim_score = gc_ms_ssim.compute().item()
        gc_psnr_score = gc_psnr.compute().item()
        dr_ms_ssim_score = dr_ms_ssim.compute().item()
        dr_psnr_score = dr_psnr.compute().item()
        avg_loss = gc_avg_loss + dr_avg_loss

        # Save to log
        self.log["val_loss"].append(dr_avg_loss)
        self.log["val_PSNR"].append(dr_psnr_score)
        self.log["val_MS_SSIM"].append(dr_ms_ssim_score)

        self.gc_log["val_loss"].append(gc_avg_loss)
        self.gc_log["val_PSNR"].append(gc_psnr_score)
        self.gc_log["val_MS_SSIM"].append(gc_ms_ssim_score)

        end = time.time()

        print(
            f"Validation Summary [{end-start:.3f}s]: \n Total Avg Loss: {avg_loss:.4f} | GC Avg Loss: {gc_avg_loss:.4f} | GC MS-SSIM: {gc_ms_ssim_score:.4f} | GC PSNR: {gc_psnr_score:.4f} | DR Avg Loss: {dr_avg_loss:.4f} | DR MS-SSIM: {dr_ms_ssim_score:.4f} | DR PSNR: {dr_psnr_score:.4f} | lr: {lr}"
        )

        return end - start

    def fit(self, train_loader, val_loader):
        if self.dr_checkpoint_exists:
            print("----> DRNet checkpoint found and loaded! <-----")

        print("-----Start Training!-----")
        early_stopper = EarlyStoppingMultiModel(patience=5, min_delta=0.001)

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
                self.log["val_loss"][-1] + self.gc_log["val_loss"][-1],
                self.model,
                self.gcnet,
                epoch + 1,
            ):
                print("Early Stopped!")
                break

        if self.save():
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
            # Save GCNet logs
            pd.DataFrame(self.gc_log).to_csv(
                f"{LOGS_PATH}/{self.gcnet.name}.csv", index=False
            )
            print("Logs saved!")

        print("-----Done Training!-----")
