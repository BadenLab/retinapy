import torch
import torch.nn as nn
from torch.nn import functional as F


class Decoder1dBlock(nn.Module):
    """
    I referred to decoder architecture here:
    https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/decoders
    """

    def __init__(self, in_channels, out_channels, act=True):
        super(Decoder1dBlock, self).__init__()
        self.act = act
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.bn1(self.conv1(x)))
        if self.act:
            x = F.relu(self.bn2(self.conv2(x)))
        else:
            x = self.conv2(x)
        return x


class Residual1dBlock(nn.Module):
    def __init__(self, in_n, mid_n, out_n, downsample=False):
        super(Residual1dBlock, self).__init__()
        self.downsample = downsample
        stride = 2 if self.downsample else 1
        self.conv1 = nn.Conv1d(
            in_n, mid_n, kernel_size=1, stride=1, padding=0, dilation=1
        )
        self.conv2 = nn.Conv1d(
            in_n, mid_n, kernel_size=5, stride=stride, padding=2, dilation=1
        )
        self.conv3 = nn.Conv1d(
            mid_n, out_n, kernel_size=1, stride=1, padding=0, dilation=1
        )
        self.bn1 = nn.BatchNorm1d(in_n)
        self.bn2 = nn.BatchNorm1d(mid_n)
        self.bn3 = nn.BatchNorm1d(mid_n)
        if downsample:
            self.shortcut_downsample = nn.Conv1d(
                in_n,
                out_n,
                kernel_size=1,
                stride=2,
                padding=0,
                dilation=1,
                bias=False,
            )
        else:
            self.shortcut_downsample = nn.Identity()

    def forward(self, x):
        shortcut = x
        shortcut = self.shortcut_downsample(shortcut)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.relu(x + shortcut)
        return x


class FcLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm([out_features]),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.net(input)


class FcBlock(nn.Module):
    def __init__(
        self,
        hidden_ch,
        num_hidden_layers,
        in_features,
        out_features,
        outermost_linear=False,
    ):
        super().__init__()

        self.net_elements = []
        self.net_elements.append(nn.Flatten())
        self.net_elements.append(
            FcLayer(in_features=in_features, out_features=hidden_ch)
        )

        for i in range(num_hidden_layers):
            self.net_elements.append(
                FcLayer(in_features=hidden_ch, out_features=hidden_ch)
            )

        if outermost_linear:
            self.net_elements.append(
                nn.Linear(in_features=hidden_ch, out_features=out_features)
            )
        else:
            self.net_elements.append(
                FcLayer(in_features=hidden_ch, out_features=out_features)
            )

        self.net = nn.Sequential(*self.net_elements)
        self.net.apply(self.init_weights)

    def __getitem__(self, item):
        return self.net_elements[item]

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(
                m.weight, a=0.0, nonlinearity="relu", mode="fan_in"
            )

    def forward(self, input):
        ans = self.net(input)
        return ans

class Conv1dSame(torch.nn.Module):
    """1D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        padding_layer=nn.ReflectionPad1d,
    ):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        """
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv1d(
                in_channels, out_channels, kernel_size, bias=bias, stride=1
            ),
        )

        self.weight = self.net[1].weight
        self.bias = self.net[1].bias

    def forward(self, x):
        return self.net(x)
