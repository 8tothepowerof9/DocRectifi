from .base import BaseModel
from .baseline import Baseline
from .unet_based import GCNet, DRNet, UNext

MODEL_LIST = {"baseline": Baseline, "unext": UNext, "gcnet": GCNet, "gcdr": DRNet}
