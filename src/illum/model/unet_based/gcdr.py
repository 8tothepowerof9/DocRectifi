from .unext import DepthWiseConv, OverlapPatchEmbed, ShiftMLP, ShiftedBlock
from ..base import BaseModel


class GCDRNet(BaseModel):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
