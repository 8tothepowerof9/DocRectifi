from .base import BaseTrainer
from .std_trainer import StandardTrainer
from .gcdr_trainer import GCDRTrainer

TRAINER_LIST = {"standard": StandardTrainer, "gcdr": GCDRTrainer}
