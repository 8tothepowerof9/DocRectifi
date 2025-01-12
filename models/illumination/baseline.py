from torch import nn


# The baseline model is a simple fully convnet


class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.name = cfg["model"]["name"]

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
