from .base import BaseTrainer
from .std_trainer import StandardTrainer
from .gcdr_trainer import GCDRTrainer, GCTrainer

TRAINER_LIST = {"standard": StandardTrainer, "gc": GCTrainer, "gcdr": GCDRTrainer}
