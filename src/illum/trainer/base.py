from abc import ABC, abstractmethod
from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)
import os
import torch
from torch import optim
from ..config import CHECKPOINTS_PATH, LOGS_PATH
import pandas as pd


class BaseTrainer(ABC):
    """
    Base class for all trainers
    """

    def __init__(
        self,
        model,
        config,
    ):
        self.model = model
        self.config = config
        self.metrics = {
            "PSNR": PSNR().to(device="cuda"),
            "MS_SSIM": MS_SSIM().to(device="cuda"),
        }  # For now use default parameters
        self.log = {
            "loss": [],
            "val_loss": [],
            "PSNR": [],
            "MS_SSIM": [],
            "val_PSNR": [],
            "val_MS_SSIM": [],
            "lr": [],
        }

    @abstractmethod
    def _train_epoch(self, dataloader):
        pass

    @abstractmethod
    def _eval_epoch(self, dataloader):
        pass

    @abstractmethod
    def fit(self, train_loader, val_loader):
        pass

    def load_checkpoint(self):
        # Find if checkpoint exists, if so, load and return True, else return False
        checkpoint_path = f"{CHECKPOINTS_PATH}/{self.config['model']['name']}.pt"
        log_path = f"{LOGS_PATH}/{self.config['model']['name']}.json"

        if os.path.exists(log_path):
            self.log = pd.read_csv(log_path)

        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            return True
        else:
            return False
