import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class SwishBackend(torch.autograd.Function):
    """Autograd implementation of Swish activation"""

    @staticmethod
    def forward(ctx, input_):
        """Forward pass

        Compute the swish activation and save the input tensor for backward
        """
        output = input_ * torch.sigmoid(input_)
        ctx.save_for_backward(input_)
        return output

    @staticmethod
    def backward(ctx, grade_output):
        """Backward pass

        Compute the gradient of Swish activation w.r.t. grade_ouput
        """
        input_ = ctx.saved_variables[0]
        i_sigmoid = torch.sigmoid(input_)
        return grade_output * (i_sigmoid * (1 + input_ * (1 - i_sigmoid)))


class Swish(nn.Module):
    """ Wrapper for Swish activation function.

    Refs:
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    def forward(self, x):
        # return x * F.sigmoid(x)
        return SwishBackend.apply(x)


def drop_connect(inputs, training=True, survival_prob=1.0):
    """Drop the entire conv with given survival probability."""
    if not training:
        return inputs

    # Compute tensor
    batch_size = inputs.shape[0]
    random_tensor = survival_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1],
                                dtype=inputs.dtype,
                                device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / survival_prob * binary_tensor
    return output


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 image_size=None, **kwargs):
        super(Conv2dSamePadding, self).__init__(in_channels,
                                                out_channels,
                                                kernel_size,
                                                **kwargs)
        self.stride = self.stride if len(self.stride) == 2 \
            else [self.stride[0]] * 2

        assert image_size is not None
        self.image_size = image_size if type(image_size) == list \
            else [image_size, image_size]
        self.out_resolution = [
            math.ceil(self.image_size[0] / self.stride[0] + 1),
            math.ceil(self.image_size[1] / self.stride[1] + 1)
        ]
        ih, iw = self.image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0]
                    + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1]
                    + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            padding_size = (pad_w // 2,
                            pad_w - pad_w // 2,
                            pad_h // 2,
                            pad_h - pad_h // 2)
            self.padding_op = nn.ZeroPad2d(padding_size)
        else:
            self.padding_op = Identity()

    def get_out_resolution(self):
        return self.out_resolution

    def forward(self, x):
        x = self.padding_op(x)
        # x = super(Conv2dSamePadding, self).forward(x)
        x = F.conv2d(x,
                     self.weight, self.bias, self.stride,
                     self.padding, self.dilation, self.groups)
        return x


class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive."""
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input
