import torch
import torch.nn as nn
from torch.nn import functional as F

import retinapy
import retinapy.nn


class DistanceFieldCnnModel(nn.Module):
    LED_CHANNELS = 4
    NUM_CLUSTERS = 1

    def __init__(self, clamp_max=None):
        super(DistanceFieldCnnModel, self).__init__()
        self.clamp_max = clamp_max
        self.num_input_channels = self.LED_CHANNELS + self.NUM_CLUSTERS
        self.num_channels = 50
        self.conv1 = nn.Conv1d(
            self.num_input_channels,
            self.num_channels,
            kernel_size=11,
            stride=2,
            padding=5,
            bias=True,
        )
        self.bn1 = nn.BatchNorm1d(self.num_channels)
        self.layer1 = nn.Sequential(
            retinapy.nn.Residual1dBlock(*(self.num_channels,) * 3, downsample=False),
            retinapy.nn.Residual1dBlock(*(self.num_channels,) * 3, downsample=False),
            retinapy.nn.Residual1dBlock(*(self.num_channels,) * 3, downsample=False),
            retinapy.nn.Residual1dBlock(*(self.num_channels,) * 3, downsample=False),
            retinapy.nn.Residual1dBlock(*(self.num_channels,) * 3, downsample=True),
            retinapy.nn.Residual1dBlock(*(self.num_channels,) * 3, downsample=False),
            retinapy.nn.Residual1dBlock(*(self.num_channels,) * 3, downsample=False),
            retinapy.nn.Residual1dBlock(*(self.num_channels,) * 3, downsample=False),
        )
        self.layer2 = nn.Sequential(
            retinapy.nn.Residual1dBlock(*(self.num_channels,) * 3, downsample=False),
            retinapy.nn.Residual1dBlock(*(self.num_channels,) * 3, downsample=True),
        )
        self.layer3 = nn.Sequential(
            retinapy.nn.Residual1dBlock(*(self.num_channels,) * 3, downsample=False),
            retinapy.nn.Residual1dBlock(*(self.num_channels,) * 3, downsample=True),
        )
        self.layer_dc1 = nn.Sequential(
            retinapy.nn.Decoder1dBlock(self.num_channels, self.num_channels),
            retinapy.nn.Residual1dBlock(*(self.num_channels,) * 3, downsample=False),
        )
        self.layer_dc2 = nn.Sequential(
            retinapy.nn.Decoder1dBlock(self.num_channels, self.num_channels),
            retinapy.nn.Residual1dBlock(*(self.num_channels,) * 3, downsample=False),
        )
        self.layer_dc3 = nn.Sequential(
            retinapy.nn.Decoder1dBlock(self.num_channels, self.num_channels),
            retinapy.nn.Residual1dBlock(*(self.num_channels,) * 3, downsample=False),
        )
        self.linear = nn.Linear(
            in_features=self.num_channels * 400, out_features=400
        )
        #self.act_fn = torch.sigmoid
        self.act_fn = torch.nn.Softplus()

    def forward(self, x):
        # 1600
        x = F.relu(self.bn1(self.conv1(x)))
        # 800
        x = self.layer1(x)
        # 400

        # Skipping for now 
        # ----------------
        #x = self.layer2(x)
        ## 200
        #x = self.layer3(x)
        ## 100
        #x = self.layer_dc1(x)
        ## 200
        #x = self.layer_dc3(x)
        ####

        x = self.linear(torch.flatten(x, start_dim=1))
        x = self.act_fn(x)
        # max=None means no clap is applied (same for min).
        x = torch.clamp(x, min=None, max=self.clamp_max)
        return x






