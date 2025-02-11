import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    StructuralSimilarityIndexMeasure as SSIM,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)
import os
import pandas as pd
from .base import BaseTrainer
from ..utils import (
    EarlyStopping,
    EarlyStoppingMultiModel,
    seconds_to_minutes_str,
    pad_to_stride,
    remove_padding,
)
from ..config import *
from ..model import GCNet
from ..loss import TVLoss, VGGLoss


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
        ms_ssim = self.metrics["MS_SSIM"]
        psnr.reset()
        ms_ssim.reset()

        self.model.train()

        for batch, (in_img, gt_img) in enumerate(dataloader):
            in_img, gt_img = in_img.to("cuda"), gt_img.to("cuda")

            # Resize to IMG_HxIMG_W
            in_img = F.interpolate(
                in_img, (IMG_H, IMG_W), mode="bilinear", align_corners=False
            )
            gt_img = F.interpolate(
                gt_img, (IMG_H, IMG_W), mode="bilinear", align_corners=False
            )

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

                # Resize to IMG_HxIMG_W
                in_img = F.interpolate(
                    in_img, (IMG_H, IMG_W), mode="bilinear", align_corners=False
                )
                gt_img = F.interpolate(
                    gt_img, (IMG_H, IMG_W), mode="bilinear", align_corners=False
                )

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
        self.lambda_3 = 0.0001

        # l8 uses L1 loss only
        # l2 and l4 has the same loss function: SSIM loss + L1 loss
        # l1 uses L1 loss + SSIM loss + Total Variation loss
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM().to("cuda")  # For now use the default values
        self.tv_loss = TVLoss(p=1)
        self.vgg_loss = VGGLoss().to("cuda")

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

        # Mixed precision gradient scaler
        scaler = GradScaler()

        # Metrics
        dr_psnr = self.metrics["PSNR"]
        dr_ms_ssim = self.metrics["MS_SSIM"]
        dr_psnr.reset()
        dr_ms_ssim.reset()

        self.model.train()
        self.gcnet.train()

        accumulation_steps = 8

        for batch, (in_img, gt_img) in enumerate(dataloader):
            in_img, gt_img = in_img.to("cuda"), gt_img.to("cuda")
            _, _, h, w = in_img.shape  # Original size

            # For source image with resolutions < 512, resize the short side to 512 while maintaining the aspect ratio
            # short_side = min(h, w)
            # if short_side < 512:
            #     scale = 512 / short_side
            #     h = int(h * scale)
            #     w = int(w * scale)
            #     in_img = F.interpolate(
            #         in_img, (h, w), mode="bilinear", align_corners=False
            #     )
            #     gt_img = F.interpolate(
            #         gt_img, (h, w), mode="bilinear", align_corners=False
            #     )

            # Resize short side of all images to 512 while maintaining aspect ratio
            short_side = min(h, w)
            scale = 512 / short_side
            h = int(h * scale)
            w = int(w * scale)
            in_img = F.interpolate(in_img, (h, w), mode="bilinear", align_corners=False)
            gt_img = F.interpolate(gt_img, (h, w), mode="bilinear", align_corners=False)

            ## Mixed Precision Training
            with autocast(device_type="cuda", dtype=torch.float16):
                ## Train GCNet
                shadow_map = torch.clamp(in_img / (gt_img + 1e-6), 0, 1)
                in_img_down = F.interpolate(in_img, (IMG_H, IMG_W), mode="bilinear")
                pred_shadow_map = self.gcnet(in_img_down)

                # Upscale the shadow map to the original size
                pred_shadow_map = F.interpolate(
                    pred_shadow_map, size=(h, w), mode="bilinear"
                )
                gc_loss = self.l1_loss(pred_shadow_map, shadow_map)
                gc_loss = gc_loss / accumulation_steps

                # Gradient of the DR-Net will not propagate back to the GC-Net
                pred_shadow_map = pred_shadow_map.detach()
                i_gc = torch.clamp(in_img / pred_shadow_map, 0, 1)

                # Ensure the size is divisible by 32 for DRNet
                in_img, padding_h, padding_w = pad_to_stride(in_img, stride=32)
                i_gc, _, _ = pad_to_stride(i_gc, stride=32)
                gt_img, _, _ = pad_to_stride(gt_img, stride=32)

                dr_input = torch.cat((in_img, i_gc), dim=1)

                ## Train DRNet
                out8, out4, out2, out1 = self.model(dr_input)

                # Remove padding from outputs and ground truth
                out8, out4, out2, out1, gt_img = (
                    remove_padding(out8, padding_h, padding_w),
                    remove_padding(out4, padding_h, padding_w),
                    remove_padding(out2, padding_h, padding_w),
                    remove_padding(out1, padding_h, padding_w),
                    remove_padding(gt_img, padding_h, padding_w),
                )

                # Resize ground truth to match outputs
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

                ## Multiscale losses
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
                dr_loss = dr_loss / accumulation_steps

            ## Backpropagation with Mixed Precision
            scaler.scale(gc_loss).backward()
            scaler.scale(dr_loss).backward()

            if (batch + 1) % accumulation_steps == 0:
                # Step the optimizers with scaled gradients
                scaler.step(self.gc_optimizer)
                scaler.step(self.dr_optimizer)

                # Update the scaler
                scaler.update()

                # Zero gradients for the combined optimizer
                self.gc_optimizer.zero_grad()
                self.dr_optimizer.zero_grad()

            # Track total loss
            dr_total_loss += dr_loss.detach().item()

            ## Update metrics
            dr_ms_ssim(out1, gt1)
            dr_psnr(out1, gt1)

            # Logging
            if batch % 100 == 0:
                with torch.no_grad():
                    dr_loss_val = dr_loss.detach().item()
                    current = batch * len(in_img)

                    print(f"loss: {dr_loss_val:.4f} [{current:>5d}/{size:>5d}]")

        lr = self.dr_optimizer.param_groups[0]["lr"]
        dr_avg_loss = dr_total_loss / num_batches
        dr_ms_ssim_score = dr_ms_ssim.compute().item()
        dr_psnr_score = dr_psnr.compute().item()

        # Save to log
        self.log["loss"].append(dr_avg_loss)
        self.log["PSNR"].append(dr_psnr_score)
        self.log["MS_SSIM"].append(dr_ms_ssim_score)
        self.log["lr"].append(lr)

        end = time.time()

        print(
            f"Train Summary [{end-start:.3f}s]: \n Avg Loss: {dr_avg_loss:.4f} | MS-SSIM: {dr_ms_ssim_score:.4f} | PSNR: {dr_psnr_score:.4f} | lr: {lr}"
        )

        return end - start

    def _eval_epoch(self, dataloader):
        start = time.time()
        num_batches = len(dataloader)
        dr_total_loss = 0

        # Metrics
        dr_psnr = self.metrics["PSNR"]
        dr_ms_ssim = self.metrics["MS_SSIM"]
        dr_psnr.reset()
        dr_ms_ssim.reset()

        self.model.eval()
        self.gcnet.eval()

        with torch.no_grad():
            for in_img, gt_img in dataloader:
                in_img, gt_img = in_img.to("cuda"), gt_img.to("cuda")
                _, _, h, w = in_img.shape  # Original size

                # Resize the short side to 512 while maintaining the aspect ratio
                # short_side = min(h, w)
                # if short_side < 512:
                #     scale = 512 / short_side
                #     h = int(h * scale)
                #     w = int(w * scale)
                #     in_img = F.interpolate(
                #         in_img, (h, w), mode="bilinear", align_corners=False
                #     )
                #     gt_img = F.interpolate(
                #         gt_img, (h, w), mode="bilinear", align_corners=False
                #     )

                short_side = min(h, w)
                scale = 512 / short_side
                h = int(h * scale)
                w = int(w * scale)
                in_img = F.interpolate(
                    in_img, (h, w), mode="bilinear", align_corners=False
                )
                gt_img = F.interpolate(
                    gt_img, (h, w), mode="bilinear", align_corners=False
                )

                # Mixed Precision Inference
                with autocast(device_type="cuda", dtype=torch.float16):
                    # GCNet
                    in_img_down = F.interpolate(in_img, (IMG_H, IMG_W), mode="bilinear")
                    pred_shadow_map = self.gcnet(in_img_down)

                    ## Upscale the shadow map to the original size
                    pred_shadow_map = F.interpolate(
                        pred_shadow_map, size=(h, w), mode="bilinear"
                    )

                    i_gc = torch.clamp(in_img / pred_shadow_map, 0, 1)
                    in_img, padding_h, padding_w = pad_to_stride(in_img, stride=32)
                    i_gc, _, _ = pad_to_stride(i_gc, stride=32)
                    gt_img, _, _ = pad_to_stride(gt_img, stride=32)

                    dr_input = torch.cat((in_img, i_gc), dim=1)

                    # DRNet
                    out8, out4, out2, out1 = self.model(dr_input)

                    out8, out4, out2, out1, gt_img = (
                        remove_padding(out8, padding_h, padding_w),
                        remove_padding(out4, padding_h, padding_w),
                        remove_padding(out2, padding_h, padding_w),
                        remove_padding(out1, padding_h, padding_w),
                        remove_padding(gt_img, padding_h, padding_w),
                    )

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

                # Store total loss
                dr_total_loss += dr_loss.item()

                # Update metrics
                dr_ms_ssim(out1, gt1)
                dr_psnr(out1, gt1)

        lr = self.dr_optimizer.param_groups[0]["lr"]
        dr_avg_loss = dr_total_loss / num_batches
        dr_ms_ssim_score = dr_ms_ssim.compute().item()
        dr_psnr_score = dr_psnr.compute().item()

        # Save to log
        self.log["val_loss"].append(dr_avg_loss)
        self.log["val_PSNR"].append(dr_psnr_score)
        self.log["val_MS_SSIM"].append(dr_ms_ssim_score)

        end = time.time()

        print(
            f"Eval Summary [{end-start:.3f}s]: \n Avg Loss: {dr_avg_loss:.4f} | MS-SSIM: {dr_ms_ssim_score:.4f} | PSNR: {dr_psnr_score:.4f} | lr: {lr}\n"
        )

        return end - start

    def fit(self, train_loader, val_loader):
        # Clear memory
        torch.cuda.empty_cache()

        if self.dr_checkpoint_exists:
            print("----> DRNet checkpoint found and loaded! <-----")

        print("-----Start Training!-----")
        early_stopper = EarlyStoppingMultiModel(patience=5, min_delta=0.01)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            _ = self._train_epoch(train_loader)
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
            # Save GCNet logs
            pd.DataFrame(self.gc_log).to_csv(
                f"{LOGS_PATH}/{self.gcnet.name}.csv", index=False
            )
            print("Logs saved!")

        print("-----Done Training!-----")
