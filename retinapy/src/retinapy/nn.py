import torch
import torch.nn as nn
from torch.nn import functional as F


def create_batch_norm(n):
    """
    Following Davis and Frank's recommendation in "Revisiting Batch
    Normazilation", we would initialize batch norm weights to less than 1 
    (they suggested to use 0.1). They also reccomended using a lower learning 
    rate for the γ parameter.

    For comparison, fastai initialize β to 0.001 and γ to 1.

    I tried both, and found better results with fastai's defaults.
    """
    bn = nn.BatchNorm1d(n)
    # fastai 
    #bn.weight.data.fill_(1.0)
    #bn.bias.data.fill_(1e-3)
    # Davis and Frank
    bn.weight.data.fill_(0.1)
    bn.bias.data.fill_(0)
    return bn


class Decoder1dBlock(nn.Module):
    """
    I referred to decoder architecture here:
    https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/decoders
    """

    def __init__(self, in_channels, out_channels, act=True):
        super(Decoder1dBlock, self).__init__()
        self.act = act
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            # No need for bias here, since we're using batch norm.
            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            # No need for bias here, since we're using batch norm.
            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            bias=False,
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


class ResBlock1d(nn.Module):
    """A residual block with 1d convolutions.

    This is a pretty inflexible implementation. No need to make it any
    more general yet.
    """

    def __init__(self, in_n, mid_n, out_n, kernel_size=3, downsample=False):
        super(ResBlock1d, self).__init__()
        self.downsample = downsample
        stride = 2 if self.downsample else 1
        self.shortcut = self.create_shortcut(in_n, out_n, stride=stride)
        # Note: bias is False for the conv layers, as they will be followed
        # by batch norm.
        self.conv1 = nn.Conv1d(
            in_n,
            mid_n,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
        )
        padding = (kernel_size - 1) // 2
        self.conv2 = nn.Conv1d(
            mid_n,
            mid_n,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            bias=False,
        )
        self.conv3 = nn.Conv1d(
            mid_n,
            out_n,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
        )
        self.bn1 = create_batch_norm(mid_n)
        self.bn2 = create_batch_norm(mid_n)
        self.bn3 = create_batch_norm(out_n)

    @staticmethod
    def create_shortcut(in_n, out_n, stride, downsample_type="pool"):
        """
        The identify path is one of those finicky bits of ResNet type networks.

        Depending on whether the input and output match in terms of channel
        count and dimension, we have the following behaviour:

        Match?
        ------

            | Channel count | Dimensions | Bevaviour                  |
            |---------------|------------|----------------------------|
            |      ✓        |     ✓      | identity                   |
            |      ✓        |     ✘      | pool or conv               |
            |      ✘        |     ✓      | 1x1 conv                   |
            |      ✘        |     ✘      | pool or conv and 1x1 conv  |


        The most interesting choice is whether to use a strided pool or a
        strided convolution to achieve the downsampling effect. It's
        interesting as implementations are split on which one to use. There
        are futher choices too, such as whether to use dilation in addition
        to strides, and whether to downsample before or after the 1x1 conv.

        Some implementations for reference:
            - fastai: https://github.com/fastai/fastai/blob/aa58b1316ad8e7a5fa2e434e15e5fe6df4f4db56/nbs/01_layers.ipynb
            - lightning: https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/self_supervised/resnets.py
            - pytorch image models: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
                 - there are two functions, downsample_conv() and
                   downsample_avg() that are used to create the downsampling for
                   the shortcut connection.
                 - uses ceil_mode=True, so a dimension n=7 would be reduced to
                   n=4, not n=3.

        My gut instinct is that swapping out pooling for convolution to achieve
        the downsample will naively achieve better results; however, the
        convolution is instrinsically more powerful (actual has paratemers) and
        so, if you care about parameter counts, then a fair comparison would
        involve reducing parameters elsewhere. Given that the whole point of
        the residual layer is to shortcircut a layer and allow gradients to
        flow easily, I think that pooling gets more theory points for staying
        true to this idea. Ironically, I think the original ResNet
        implementation might have used convolution.
        """
        # downsample_types = {"pool", "conv"}
        # Actually, let's just worry about pool for the moment.
        downsample_types = {"pool"}
        if downsample_type not in downsample_types:
            raise ValueError(
                f"Invalid downsample type ({downsample_type}). Expected one "
                f"of ({downsample_types})."
            )
        # 1. Identity
        # In the simplest case, we can just return the input. For this to work,
        # both the channel count and channel dimensions of the input and ouput
        # must match. The output channel dimension is determined by the stride.
        channels_match = in_n == out_n
        downsample = stride > 1
        if channels_match and not downsample:
            return nn.Identity()

        skip_layers = []
        # 2. Downsample
        if downsample:
            # The following (1, 7) input:
            # |  1  |  2  |  3  |  4  |  5  |  6  |  7  |
            # when pooled with stride=kernel=2 becomes (1, 4):
            # |    1.5    |    3.5    |    5.5   | 7 |
            pool = nn.AvgPool1d(
                kernel_size=stride,
                stride=stride,
                count_include_pad=False,
                ceil_mode=True,
            )
            skip_layers.append(pool)
        # 3. 1x1 conv
        if not channels_match:
            # There isn't a consensus on whether:
            #   - to use batch norm or a bias or neither.
            conv = nn.Conv1d(in_n, out_n, kernel_size=1, bias=False)
            skip_layers.append(conv)
        res = nn.Sequential(*skip_layers)
        return res

    def forward(self, x):
        shortcut = self.shortcut(x)
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
