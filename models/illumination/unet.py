import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.name = cfg["model"]["name"]
        self.feats = cfg["model"]["features"]

        self.enc1 = self._block(3, self.feats)
        self.enc2 = self._block(self.feats, self.feats * 2)
        self.enc3 = self._block(self.feats * 2, self.feats * 4)
        self.enc4 = self._block(self.feats * 4, self.feats * 8)

        self.bottleneck = self._block(self.feats * 8, self.feats * 16)

        self.up4 = nn.ConvTranspose2d(
            self.feats * 16, self.feats * 8, kernel_size=2, stride=2
        )
        self.dec4 = self._block(self.feats * 16, self.feats * 8)

        self.up3 = nn.ConvTranspose2d(
            self.feats * 8, self.feats * 4, kernel_size=2, stride=2
        )
        self.dec3 = self._block(self.feats * 8, self.feats * 4)

        self.up2 = nn.ConvTranspose2d(
            self.feats * 4, self.feats * 2, kernel_size=2, stride=2
        )
        self.dec2 = self._block(self.feats * 4, self.feats * 2)

        self.up1 = nn.ConvTranspose2d(
            self.feats * 2, self.feats, kernel_size=2, stride=2
        )
        self.dec1 = self._block(self.feats * 2, self.feats)

        # Output
        self.out = nn.Conv2d(self.feats, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoders
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2, stride=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2, stride=2))

        # Decoders
        dec4 = self.dec4(torch.cat((self.up4(bottleneck), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self.up3(dec4), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.up2(dec3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.up1(dec2), enc1), dim=1))

        return self.sigmoid(self.out(dec1))

    @staticmethod
    def _block(in_channels, features):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )
