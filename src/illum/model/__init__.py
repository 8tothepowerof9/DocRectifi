from .base import BaseModel
from .baseline import Baseline
from .unet_based import GCDRNet, UNext

MODEL_LIST = {"baseline": Baseline, "unext": UNext, "gcdnet": GCDRNet}
