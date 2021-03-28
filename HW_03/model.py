import math
import torch
import torch.nn as nn
from model_util import Swish
from model_util import drop_connect
from model_util import Conv2dSamePadding


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block

    The plane shape of output tensor should be same as that of input tensor
    """
    def __init__(self, in_channels, squeezed_channels):
        super(SEBlock, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.se_reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=squeezed_channels,
            kernel_size=1)
        self.act_op = Swish()
        self.se_expand = nn.Conv2d(
            in_channels=squeezed_channels,
            out_channels=in_channels,
            kernel_size=1)

    def forward(self, x):
        out = self.pooling(x)
        out = self.se_expand(self.act_op(self.se_reduce(out)))
        return torch.sigmoid(out) * x


class MBConvBlock(nn.Module):
    """A class of MBConv: Mobile Inverted Residual Bottleneck."""

    def __init__(self,
                 input_filters,
                 output_filters,
                 image_size,
                 kernel_size=3,
                 stride=1,
                 expand_ratio=1,
                 id_skip=True,
                 se_ratio=None,
                 bn_mom=0.01,
                 bn_eps=1e-5):
        super(MBConvBlock, self).__init__()
        self.expand_ratio = expand_ratio
        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        self.id_skip = id_skip and ((stride == 1) and
                                    (input_filters == output_filters))

        # Expansion phase
        inp = input_filters
        oup = input_filters * expand_ratio
        self.expand_conv = nn.Conv2d(
            in_channels=inp,
            out_channels=oup,
            kernel_size=1,
            padding=0,
            bias=False)
        self.bn0 = nn.BatchNorm2d(
            num_features=oup,
            momentum=bn_mom,
            eps=bn_eps)

        # Depthwise convolution phase
        self.dw_conv = Conv2dSamePadding(
            in_channels=oup,
            out_channels=oup,
            image_size=image_size,
            groups=oup,
            kernel_size=kernel_size,
            stride=stride,
            bias=False)
        self.bn1 = nn.BatchNorm2d(
            num_features=oup,
            momentum=bn_mom,
            eps=bn_eps)

        # Squeeze and Excitation
        if self.has_se:
            num_squeezed_channels = max(1, int(input_filters * se_ratio))
            self.se_block = SEBlock(
                in_channels=oup,
                squeezed_channels=num_squeezed_channels)

        # Output phase
        self.project_conv = nn.Conv2d(
            in_channels=oup,
            out_channels=output_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn2 = nn.BatchNorm2d(
            num_features=output_filters,
            momentum=bn_mom,
            eps=bn_eps)
        self.swish = Swish()

        # Get the out resolution
        self.out_resolution = self.dw_conv.get_out_resolution()

    def get_out_resolution(self):
        return self.out_resolution

    def forward(self, inputs, survival_prob=None):
        x = inputs
        # Expansion phase (skip if expand ratio is 0)
        if self.expand_ratio != 1:
            x = self.swish(self.bn0(self.expand_conv(x)))

        # Depthwise convolution phase
        x = self.swish(self.bn1(self.dw_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x = self.se_block(x)

        # Output phase
        x = self.bn2(self.project_conv(x))

        if self.id_skip:
            if survival_prob:
                drop_connect(x, self.training, survival_prob)
            x = x + inputs
        return x


DEFAULT_BLOCKS_ARGS = [
    {"kernel_size": 3, "repeats": 1, "input_filters": 32,
     "output_filters": 16, "expand_ratio": 1,
     "id_skip": True, "strides": 1, "se_ratio": 0.25},
    {"kernel_size": 3, "repeats": 2, "input_filters": 16,
     "output_filters": 24, "expand_ratio": 6,
     "id_skip": True, "strides": 2, "se_ratio": 0.25},
    {"kernel_size": 5, "repeats": 2, "input_filters": 24,
     "output_filters": 40, "expand_ratio": 6,
     "id_skip": True, "strides": 2, "se_ratio": 0.25},
    {"kernel_size": 3, "repeats": 3, "input_filters": 40,
     "output_filters": 80, "expand_ratio": 6,
     "id_skip": True, "strides": 2, "se_ratio": 0.25},
    {"kernel_size": 5, "repeats": 3, "input_filters": 80,
     "output_filters": 112, "expand_ratio": 6,
     "id_skip": True, "strides": 1, "se_ratio": 0.25},
    {"kernel_size": 5, "repeats": 4, "input_filters": 112,
     "output_filters": 192, "expand_ratio": 6,
     "id_skip": True, "strides": 2, "se_ratio": 0.25},
    {"kernel_size": 3, "repeats": 1, "input_filters": 192,
     "output_filters": 320, "expand_ratio": 6,
     "id_skip": True, "strides": 1, "se_ratio": 0.25}
]


MINI_BLOCKS_ARGS = [
    {"kernel_size": 3, "repeats": 1, "input_filters": 32,
     "output_filters": 16, "expand_ratio": 1,
     "id_skip": True, "strides": 1, "se_ratio": 0.25},
    {"kernel_size": 3, "repeats": 1, "input_filters": 16,
     "output_filters": 24, "expand_ratio": 6,
     "id_skip": True, "strides": 2, "se_ratio": 0.25},
    {"kernel_size": 5, "repeats": 1, "input_filters": 24,
     "output_filters": 40, "expand_ratio": 6,
     "id_skip": True, "strides": 2, "se_ratio": 0.25},
    {"kernel_size": 3, "repeats": 1, "input_filters": 40,
     "output_filters": 80, "expand_ratio": 6,
     "id_skip": True, "strides": 2, "se_ratio": 0.25},
    {"kernel_size": 5, "repeats": 1, "input_filters": 80,
     "output_filters": 112, "expand_ratio": 6,
     "id_skip": True, "strides": 1, "se_ratio": 0.25},
    {"kernel_size": 5, "repeats": 1, "input_filters": 112,
     "output_filters": 192, "expand_ratio": 6,
     "id_skip": True, "strides": 2, "se_ratio": 0.25},
    {"kernel_size": 3, "repeats": 1, "input_filters": 192,
     "output_filters": 320, "expand_ratio": 6,
     "id_skip": True, "strides": 1, "se_ratio": 0.25}
]


class EfficientNet(nn.Module):
    """A class implements nn.Module for MNAS-like model.

    Reference: https://arxiv.org/abs/1807.11626
    """
    def __init__(self,
                 width_coefficient,
                 depth_coefficient,
                 image_size,
                 dropout_rate=0.2,
                 survival_prob=0.8,
                 depth_divisor=8,
                 min_depth=None,
                 batchnorm_momentum=0.99,
                 batchnorm_eps=1e-5,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 in_channels=3,
                 num_classes=10):
        super(EfficientNet, self).__init__()
        # Global params
        self.image_size = [image_size, image_size] \
            if isinstance(image_size, int) else image_size

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.survival_prob = survival_prob
        self.bn_mom = 1 - batchnorm_momentum
        self.bn_eps = batchnorm_eps

        # MBConv blocks arguments
        self.blocks_args = blocks_args

        # Build stem
        oup = self.round_filters(filters=32)
        self.conv_stem = Conv2dSamePadding(
            in_channels=in_channels,
            out_channels=oup,
            kernel_size=3,
            stride=2,
            image_size=image_size,
            bias=False)
        self.resolution = self.conv_stem.get_out_resolution()
        self.bn0 = nn.BatchNorm2d(
            num_features=oup,
            momentum=self.bn_mom,
            eps=self.bn_eps)

        # Make MBConv blocks
        self.blocks, oup = self._make_layers(in_channels=oup)

        # Head part
        in_channels = oup
        out_channels = self.round_filters(1280)
        self.conv_head = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=self.bn_mom,
            eps=self.bn_eps)

        # Fully connected layer
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(out_channels, num_classes)
        self.swish = Swish()

    def round_filters(self, filters: int, skip=False):
        """Round number of filters based on depth multiplier."""
        multiplier = self.width_coefficient
        if skip or not multiplier:
            return filters

        divisor = self.depth_divisor
        min_depth = self.min_depth
        filters *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth,
                          int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(self, repeats: int, skip=False):
        """Round number of filters based on depth multiplier."""
        multiplier = self.depth_coefficient
        if skip or not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _make_layers(self, in_channels):
        """Make the MBConv blocks according to config"""
        mbconv_blocks = nn.ModuleList([])

        for block_args in self.blocks_args:
            repeats = self.round_repeats(block_args["repeats"])
            out_channels = self.round_filters(block_args["output_filters"])
            for added in range(repeats):
                stride = block_args["strides"] if added == 0 else 1
                mbconv_block = MBConvBlock(
                    input_filters=in_channels,
                    output_filters=out_channels,
                    image_size=self.resolution,
                    kernel_size=block_args["kernel_size"],
                    stride=stride,
                    expand_ratio=block_args["expand_ratio"],
                    id_skip=block_args["id_skip"],
                    se_ratio=block_args["se_ratio"],
                    bn_mom=self.bn_mom,
                    bn_eps=self.bn_eps)

                mbconv_blocks.append(mbconv_block)
                self.resolution = mbconv_block.get_out_resolution()
                in_channels = out_channels

        return mbconv_blocks, out_channels

    def extract_features(self, inputs):
        """Feed inputs into conv layer and blocks"""
        # Stem phase
        x = self.swish(self.bn0(self.conv_stem(inputs)))

        # MBConv Blocks
        n_blocks = len(self.blocks)
        for idx, block in enumerate(self.blocks):
            survival_prob = self.survival_prob
            if survival_prob:
                survival_prob = 1 - \
                    (1 - survival_prob) * float(idx) / n_blocks
            x = block(x, survival_prob=survival_prob)
            # print(idx, x.shape)

        # Head part
        x = self.swish(self.bn1(self.conv_head(x)))
        return x

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        x = self.extract_features(inputs)

        # Pooling and fully connected layer
        x = self.pooling(x)
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def EfficientNetB0(in_channels=3, classes=1000):
    return EfficientNet(1.0, 1.0, 224, 0.2,
                        in_channels=in_channels,
                        num_classes=classes)


def EfficientNetB1(in_channels=3, classes=1000):
    return EfficientNet(1.0, 1.1, 240, 0.2,
                        in_channels=in_channels,
                        num_classes=classes)


def EfficientNetB2(in_channels=3, classes=1000):
    return EfficientNet(1.1, 1.2, 260, 0.3,
                        in_channels=in_channels,
                        num_classes=classes)


def EfficientNetB3(in_channels=3, classes=1000):
    return EfficientNet(1.2, 1.4, 300, 0.3,
                        in_channels=in_channels,
                        num_classes=classes)


def EfficientNetB4(in_channels=3, classes=1000):
    return EfficientNet(1.4, 1.8, 380, 0.4,
                        in_channels=in_channels,
                        num_classes=classes)


def EfficientNetB5(in_channels=3, classes=1000):
    return EfficientNet(1.6, 2.2, 456, 0.4,
                        in_channels=in_channels,
                        num_classes=classes)


def EfficientNetB6(in_channels=3, classes=1000):
    return EfficientNet(1.8, 2.6, 528, 0.5,
                        in_channels=in_channels,
                        num_classes=classes)


def EfficientNetB7(in_channels=3, classes=1000):
    return EfficientNet(2.0, 3.1, 600, 0.5,
                        in_channels=in_channels,
                        num_classes=classes)
