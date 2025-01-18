from torch import nn
from abc import abstractmethod


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    def __str__(self):
        return super().__str__()
