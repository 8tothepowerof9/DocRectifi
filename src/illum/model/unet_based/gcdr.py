"""
Essentially, GCDRNet is made up of 2 UNext models (with slight modification to the original UNext implementation which is in the unext.py file).
The first UNext is called GCNet, and the second UNext is called DRNet. 
GCNet implementation uses the following modifications:
- ReflectionPad2d in the encoder and decoder layers to handle padding, this ensures that feature maps maintain full resolution and spatial information without cropping or losing edges. 
The original implementation uses padding=1 which can introduce artifacts near the image edges
DRNet implmementation uses the following modifications:
- Similar to GCNet, ReflectionPad2d is also used in the encoder and decoder layers to handle padding.
- DRNet has multiple output heads, each responsible for handling different resolution, enabling deep supervision
"""

import torch
from torch import nn
from torchinfo import summary
import torch.nn.functional as F
from .unext import OverlapPatchEmbed, ShiftedBlock
from ..base import BaseModel
from ...config import IMG_H, IMG_W


class GCNet(BaseModel):
    def __init__(self, cfg):
        super().__init__()

        self.name = "GCNet"
        drop_rate = cfg["model"]["drop_rate"]
        drop_path_rate = cfg["model"]["drop_path_rate"]
        depths = cfg["model"]["depths"]
        embed_dims = cfg["model"]["embed_dims"]

        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(3, 16, 3, stride=1, padding=0)
        )
        self.encoder2 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(16, 32, 3, stride=1, padding=0)
        )
        self.encoder3 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(32, 128, 3, stride=1, padding=0)
        )

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = nn.LayerNorm(embed_dims[1])
        self.norm4 = nn.LayerNorm(embed_dims[2])

        self.dnorm3 = nn.LayerNorm(160)
        self.dnorm4 = nn.LayerNorm(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList(
            [
                ShiftedBlock(
                    dim=embed_dims[1],
                    mlp_ratio=1,
                    drop=drop_rate,
                    drop_path=dpr[0],
                )
            ]
        )

        self.block2 = nn.ModuleList(
            [
                ShiftedBlock(
                    dim=embed_dims[2],
                    mlp_ratio=1,
                    drop=drop_rate,
                    drop_path=dpr[1],
                )
            ]
        )

        self.dblock1 = nn.ModuleList(
            [
                ShiftedBlock(
                    dim=embed_dims[1],
                    mlp_ratio=1,
                    drop=drop_rate,
                    drop_path=dpr[0],
                )
            ]
        )

        self.dblock2 = nn.ModuleList(
            [
                ShiftedBlock(
                    dim=embed_dims[0],
                    mlp_ratio=1,
                    drop=drop_rate,
                    drop_path=dpr[1],
                )
            ]
        )

        self.patch_embed3 = OverlapPatchEmbed(
            img_size=IMG_W // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=IMG_H // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )

        self.decoder1 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(256, 160, 3, stride=1, padding=0)
        )
        self.decoder2 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(160, 128, 3, stride=1, padding=0)
        )
        self.decoder3 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(128, 32, 3, stride=1, padding=0)
        )
        self.decoder4 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(32, 16, 3, stride=1, padding=0)
        )
        self.decoder5 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(16, 16, 3, stride=1, padding=0)
        )

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage
        ### Stage 1
        temp = self.ebn1(self.encoder1(x))
        out = F.relu(F.max_pool2d(temp, 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(
            F.interpolate(
                self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode="bilinear"
            )
        )

        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(
            F.interpolate(
                self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode="bilinear"
            )
        )
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(
            F.interpolate(
                self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode="bilinear"
            )
        )
        out = torch.add(out, t2)
        out = F.relu(
            F.interpolate(
                self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode="bilinear"
            )
        )
        out = torch.add(out, t1)
        out = F.relu(
            F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode="bilinear")
        )

        out = torch.add(out, F.relu(temp))
        out = self.final(out)
        return self.sigmoid(out)

    def __str__(self):
        return str(summary(self, input_size=(1, 3, IMG_H, IMG_W)))


# TODO: Add GCNet before DRNet
class DRNet(BaseModel):
    def __init__(self, cfg):
        super().__init__()

        self.name = cfg["model"]["name"]
        drop_rate = cfg["model"]["drop_rate"]
        drop_path_rate = cfg["model"]["drop_path_rate"]
        depths = cfg["model"]["depths"]
        embed_dims = cfg["model"]["embed_dims"]

        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(3, 32, 3, stride=1, padding=0)
        )
        self.encoder2 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(32, 64, 3, stride=1, padding=0)
        )
        self.encoder3 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(64, 128, 3, stride=1, padding=0)
        )

        self.ebn1 = nn.BatchNorm2d(32)
        self.ebn2 = nn.BatchNorm2d(64)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = nn.LayerNorm(embed_dims[1])
        self.norm4 = nn.LayerNorm(embed_dims[2])

        self.dnorm3 = nn.LayerNorm(256)
        self.dnorm4 = nn.LayerNorm(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList(
            [
                ShiftedBlock(
                    dim=embed_dims[1],
                    mlp_ratio=1,
                    drop=drop_rate,
                    drop_path=dpr[0],
                )
            ]
        )

        self.block2 = nn.ModuleList(
            [
                ShiftedBlock(
                    dim=embed_dims[2],
                    mlp_ratio=1,
                    drop=drop_rate,
                    drop_path=dpr[1],
                )
            ]
        )

        self.dblock1 = nn.ModuleList(
            [
                ShiftedBlock(
                    dim=embed_dims[1],
                    mlp_ratio=1,
                    drop=drop_rate,
                    drop_path=dpr[0],
                )
            ]
        )

        self.dblock2 = nn.ModuleList(
            [
                ShiftedBlock(
                    dim=embed_dims[0],
                    mlp_ratio=1,
                    drop=drop_rate,
                    drop_path=dpr[1],
                )
            ]
        )

        self.patch_embed3 = OverlapPatchEmbed(
            img_size=IMG_W // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=IMG_H // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )

        self.decoder1 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3, stride=1, padding=0)
        )
        self.decoder2 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3, stride=1, padding=0)
        )
        self.decoder3 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3, stride=1, padding=0)
        )
        self.decoder4 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(64, 32, 3, stride=1, padding=0)
        )
        self.decoder5 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(32, 32, 3, stride=1, padding=0)
        )

        self.dbn1 = nn.BatchNorm2d(256)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dbn4 = nn.BatchNorm2d(32)
        self.dbn5 = nn.BatchNorm2d(32)

        self.final = nn.Conv2d(32, 3, kernel_size=1)

        self.out8 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=1),
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B = x.shape[0]
        ### Encoder
        ### Conv Stage
        ### Stage 1
        temp = self.ebn1(self.encoder1(x))
        out = F.relu(F.max_pool2d(temp, 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(
            F.interpolate(
                self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode="bilinear"
            )
        )

        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(
            F.interpolate(
                self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode="bilinear"
            )
        )
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out8 = self.sigmoid(self.out8(out))

        out = F.relu(
            F.interpolate(
                self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode="bilinear"
            )
        )
        out = torch.add(out, t2)

        out4 = self.sigmoid(self.out4(out))

        out = F.relu(
            F.interpolate(
                self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode="bilinear"
            )
        )
        out = torch.add(out, t1)

        out2 = self.sigmoid(self.out2(out))

        out = F.relu(
            F.interpolate(
                self.dbn5(self.decoder5(out)), scale_factor=(2, 2), mode="bilinear"
            )
        )
        out = torch.add(out, F.relu(temp))

        return self.sigmoid(self.final(out)), out2, out4, out8

    def __str__(self):
        return str(summary(self, input_size=(1, 3, IMG_H, IMG_W)))
