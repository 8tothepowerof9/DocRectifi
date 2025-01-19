import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from torch.utils.data import DataLoader
from illum.dataset import RealDAE
from illum.evaluator import Evaluator
from utils import read_cfg

if __name__ == "__main__":

    cfg_file = sys.argv[1]
    config = read_cfg(cfg_file)

    val_ds = RealDAE(split="val")
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    evaluator = Evaluator(config)

    if config["eval"]["vis_preds"]:
        evaluator.vis_preds(val_loader)

    if config["eval"]["plot_history"]:
        evaluator.plot_history()
