from .base import BaseModel
from .baseline import Baseline
from .unet_based import GCNet, DRNett, UNext

MODEL_LIST = {"baseline": Baseline, "unext": UNext, "gcnet": GCNet, "drnett": DRNett}
