from abc import ABC, abstractmethod
from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)


class BaseTrainer(ABC):
    """
    Base class for all trainers
    """

    def __init__(self, model, optimizer, epochs, scheduler=None, save=True):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.scheduler = scheduler
        self.save = save
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
