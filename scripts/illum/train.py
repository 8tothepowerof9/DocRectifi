import sys
import os
from utils import read_cfg

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"  # Disable version check
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from torch.utils.data import DataLoader
from illum.dataset import RealDAE, FullResBatchSampler
from illum.model import *
from illum.utils import *
from illum.trainer import *

if __name__ == "__main__":
    cfg_file = sys.argv[1]
    config = read_cfg(cfg_file)

    train_ds = RealDAE(split="train", min_mem_usage=True)
    val_ds = RealDAE(split="val", min_mem_usage=True)

    # Batch sampler
    train_sampler = FullResBatchSampler(
        config["data"]["batch_size"], train_ds.imgs_size_idx, shuffle=True
    )
    val_sampler = FullResBatchSampler(
        config["data"]["batch_size"], val_ds.imgs_size_idx, shuffle=False
    )

    train_loader = DataLoader(
        train_ds,
        num_workers=config["data"]["num_workers"],
        batch_sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_ds,
        num_workers=config["data"]["num_workers"],
        batch_sampler=val_sampler,
    )

    # Get model
    # This will create DRNet if load gcdr
    model = MODEL_LIST[config["model"]["type"]](config).to("cuda")

    if config["train"]["trainer"] not in TRAINER_LIST.keys():
        raise ValueError("Invalid trainer type")

    # Wil automatically load gcnet from config file
    trainer = TRAINER_LIST[config["train"]["trainer"]](
        model=model,
        config=config,
    )

    trainer.fit(train_loader, train_loader)
