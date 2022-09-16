import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union
import pathlib

import retinapy
import retinapy.nn

import logging

_logger = logging.getLogger(__name__)


def load_model(model, checkpoint_path: Union[str, pathlib.Path]):
    checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file/folder ({checkpoint_path}) not found."
        )
    if checkpoint_path.is_dir():
        checkpoint_path = list(checkpoint_path.glob("*.pth"))[-1]

    _logger.info(f"Loading model from ({checkpoint_path}).")
    checkpoint_state = torch.load(checkpoint_path)
    model_state = checkpoint_state["model"]
    model.load_state_dict(model_state)


def save_model(model, path: pathlib.Path, optimizer=None):
    _logger.info(f"Saving model to ({path})")
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    state = {
        "model": model.state_dict(),
    }
    if optimizer:
        state.update({"optimizer": optimizer.state_dict()})
    torch.save(state, path)


class DistLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.01):
        super(DistLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        error = F.l1_loss(prediction, target)
        scaling = self.alpha + 1 / (self.beta + target)
        loss = torch.mean(error * scaling)
        return loss


class IntervalWeightedDistLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.05):
        super(IntervalWeightedDistLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target, interval):
        error = F.l1_loss(prediction, target)
        scaling = self.alpha + 1 / (self.beta + interval)
        loss = torch.mean(error * scaling)
        return loss


class FcModel(nn.Module):
    def __init__(self, in_n, out_n, clamp_max):
        super(FcModel, self).__init__()
        self.clamp_max = clamp_max
        self.fc_body = retinapy.nn.FcBlock(
            hidden_ch=1024 * 2,
            num_hidden_layers=5,
            in_features=in_n,
            out_features=out_n,
            outermost_linear=True,
        )
        self.act = torch.nn.Softplus()

    def forward(self, x):
        x = self.act(self.fc_body(x))
        x = torch.clamp(x, min=None, max=self.clamp_max)
        return x


class DistanceFieldCnnModel(nn.Module):
    LED_CHANNELS = 4
    NUM_CLUSTERS = 1

    def __init__(self, clamp_max, in_len, out_len, num_halves):
        super(DistanceFieldCnnModel, self).__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.num_halves = num_halves
        self.num_input_channels = self.LED_CHANNELS + self.NUM_CLUSTERS
        self.clamp_max = clamp_max
        self.num_channels = 60
        self.l1_num_channels = 60
        kernel_size = 101
        mid_kernel_size = 11
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                self.num_input_channels,
                self.l1_num_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                bias=True,
            ),
            nn.Conv1d(
                self.l1_num_channels,
                self.l1_num_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=True,
            ),
            nn.Conv1d(
                self.l1_num_channels,
                self.l1_num_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=True,
            ),
        )
        self.bn1 = nn.BatchNorm1d(self.l1_num_channels)
        self.layer1 = nn.Sequential(
            retinapy.nn.Residual1dBlock(
                self.l1_num_channels,
                self.num_channels,
                self.num_channels,
                kernel_size=mid_kernel_size,
                downsample=False,
            ),
            *[
                retinapy.nn.Residual1dBlock(
                    *(self.num_channels,) * 3,
                    kernel_size=mid_kernel_size,
                    downsample=True,
                )
                for n in range(num_halves - 1)
            ],
        )
        linear_in_len = 1 + (in_len - 1) // 2**num_halves
        self.linear = nn.Linear(
            in_features=self.num_channels * linear_in_len,
            out_features=self.out_len,
        )
        # self.act_fn = torch.sigmoid
        self.act_fn = torch.nn.Softplus()

    def forward(self, x):
        # L
        x = F.relu(self.bn1(self.conv1(x)))
        # L//2
        x = self.layer1(x)
        # L//2**num_halves
        x = self.linear(torch.flatten(x, start_dim=1))
        x = self.act_fn(x)
        # max=None means no clap is applied (same for min).
        x = torch.clamp(x, min=None, max=self.clamp_max)
        return x


class DistanceFieldCnnModelOld(nn.Module):
    LED_CHANNELS = 4
    NUM_CLUSTERS = 1

    def __init__(self, clamp_max):
        super(DistanceFieldCnnModelOld, self).__init__()
        self.num_input_channels = self.LED_CHANNELS + self.NUM_CLUSTERS
        self.num_channels = 60
        self.clamp_max = clamp_max
        self.l1_num_channels = 60
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                self.num_input_channels,
                self.l1_num_channels,
                kernel_size=101,
                stride=2,
                padding=101 // 2,
                bias=True,
            ),
            nn.Conv1d(
                self.l1_num_channels,
                self.l1_num_channels,
                kernel_size=101,
                stride=1,
                padding=101 // 2,
                bias=True,
            ),
            nn.Conv1d(
                self.l1_num_channels,
                self.l1_num_channels,
                kernel_size=101,
                stride=1,
                padding=101 // 2,
                bias=True,
            ),
        )
        k = 11
        self.bn1 = nn.BatchNorm1d(self.l1_num_channels)
        self.layer1 = nn.Sequential(
            retinapy.nn.Residual1dBlock(
                self.l1_num_channels,
                self.num_channels,
                self.num_channels,
                kernel_size=k,
                downsample=False,
            ),
            retinapy.nn.Residual1dBlock(
                *(self.num_channels,) * 3, kernel_size=k, downsample=False
            ),
            retinapy.nn.Residual1dBlock(
                *(self.num_channels,) * 3, kernel_size=k, downsample=False
            ),
            retinapy.nn.Residual1dBlock(
                *(self.num_channels,) * 3, kernel_size=k, downsample=False
            ),
            retinapy.nn.Residual1dBlock(
                *(self.num_channels,) * 3, kernel_size=k, downsample=True
            ),
            retinapy.nn.Residual1dBlock(
                *(self.num_channels,) * 3, kernel_size=k, downsample=False
            ),
            retinapy.nn.Residual1dBlock(
                *(self.num_channels,) * 3, kernel_size=k, downsample=False
            ),
            retinapy.nn.Residual1dBlock(
                *(self.num_channels,) * 3, kernel_size=k, downsample=False
            ),
        )
        self.layer2 = nn.Sequential(
            retinapy.nn.Residual1dBlock(
                *(self.num_channels,) * 3, kernel_size=k, downsample=False
            ),
            retinapy.nn.Residual1dBlock(
                *(self.num_channels,) * 3, kernel_size=k, downsample=True
            ),
        )
        self.layer3 = nn.Sequential(
            retinapy.nn.Residual1dBlock(
                *(self.num_channels,) * 3, kernel_size=k, downsample=False
            ),
            retinapy.nn.Residual1dBlock(
                *(self.num_channels,) * 3, kernel_size=k, downsample=True
            ),
        )
        self.layer_dc1 = nn.Sequential(
            retinapy.nn.Decoder1dBlock(self.num_channels, self.num_channels),
            retinapy.nn.Residual1dBlock(
                *(self.num_channels,) * 3, kernel_size=k, downsample=False
            ),
        )
        self.layer_dc2 = nn.Sequential(
            retinapy.nn.Decoder1dBlock(self.num_channels, self.num_channels),
            retinapy.nn.Residual1dBlock(
                *(self.num_channels,) * 3, kernel_size=k, downsample=False
            ),
        )
        self.layer_dc3 = nn.Sequential(
            retinapy.nn.Decoder1dBlock(self.num_channels, self.num_channels),
            retinapy.nn.Residual1dBlock(
                *(self.num_channels,) * 3, kernel_size=k, downsample=False
            ),
        )
        self.linear = nn.Linear(
            in_features=self.num_channels * 400, out_features=400
        )
        # self.act_fn = torch.sigmoid
        self.act_fn = torch.nn.Softplus()

    def forward(self, x):
        # 1600
        x = F.relu(self.bn1(self.conv1(x)))
        # 800
        x = self.layer1(x)
        # 400

        # Skipping for now
        # ----------------
        # x = self.layer2(x)
        ## 200
        # x = self.layer3(x)
        ## 100
        # x = self.layer_dc1(x)
        ## 200
        # x = self.layer_dc3(x)
        ####

        x = self.linear(torch.flatten(x, start_dim=1))
        x = self.act_fn(x)
        # max=None means no clap is applied (same for min).
        x = torch.clamp(x, min=None, max=self.clamp_max)
        return x


class DistFieldToSpikeModel(nn.Module):
    def __init__(self, in_len):
        super(DistFieldToSpikeModel, self).__init__()
        self.act = nn.Softplus()
        k = 5
        c = 20
        n = 3
        self.layer1 = nn.Sequential(
            retinapy.nn.Residual1dBlock(
                1, c, c, kernel_size=k, downsample=True
            ),
            *[
                retinapy.nn.Residual1dBlock(
                    c,
                    c,
                    c,
                    kernel_size=k,
                    downsample=True,
                )
                for _ in range(n - 1)
            ],
        )
        linear_in_len = (in_len // 2**n) * c
        self.linear = nn.Linear(in_features=linear_in_len, out_features=1)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.layer1(x)
        x = self.linear(torch.flatten(x, start_dim=1))
        x = self.act(x)
        x = torch.squeeze(x, dim=1)
        return x


class LinearNonlinear(nn.Module):
    # TODO: add options for 'rbf' and  'psp' activations.

    def __init__(self, in_n, out_n):
        super(LinearNonlinear, self).__init__()
        self.linear = nn.Linear(in_features=in_n, out_features=out_n)
        self.act = nn.Softplus()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = self.act(x)
        # Remove the last dimension, as it is always 1.
        assert x.shape[1] == 1
        x = torch.squeeze(x, dim=1)
        return x


class DeepRetina2016(nn.Module):
    """
    From:
        https://github.com/baccuslab/deep-retina/blob/master/deepretina/models.py
    """

    def __init__(self, in_len, in_n, out_n):
        super(DeepRetina2016, self).__init__()
        self.in_n = in_n
        self.out_n = out_n
        self.in_len = in_len

        self.conv1 = nn.Conv2d(
            in_channels=self.in_n, out_channels=16, kernel_size=15
        )
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=9)
        linear_in_len = in_len - 15 + 1 - 9 + 1
        self.linear = nn.Linear(linear_in_len, self.out_n)
        self.act = nn.Softplus()
        # self.linear = nn.Linear(in_features=?, out_features=self.num_softmax_classes)

    def forward(self, x):
        # TODO: add noise?
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = self.act(x)
