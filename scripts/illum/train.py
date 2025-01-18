import sys
import os
from utils import read_cfg
import torch
from torch import nn, optim

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"  # Disable version check
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from torch.utils.data import DataLoader
from illum.dataset import RealDAE
from illum.model import *
from illum.utils import *
from illum.trainer import *

if __name__ == "__main__":
    cfg_file = sys.argv[1]
    config = read_cfg(cfg_file)

    device = "cuda"

    train_ds = RealDAE(split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    val_ds = RealDAE(split="val")
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    # Get model
    model = MODEL_LIST[config["model"]["type"]](config).to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

    if config["train"]["trainer"] not in TRAINER_LIST.keys():
        raise ValueError("Invalid trainer type")

    trainer = TRAINER_LIST[config["train"]["trainer"]](
        model=model,
        loss_fn=nn.L1Loss(),
        optimizer=optimizer,
        scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5),
        save=config["train"]["save"],
    )

    trainer.fit(train_loader, val_loader, config["train"]["epochs"])
