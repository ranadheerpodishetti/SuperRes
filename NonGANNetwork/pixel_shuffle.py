#Author: Soumick Chatterjee

import torch
from torch import nn
import torchvision
import math

from collections import namedtuple

def pixel_shuffle_generic(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(N, C, d_{1}, d_{2}, ..., d_{n})` to a
    tensor of shape :math:`(N, C/(r^n), d_{1}*r, d_{2}*r, ..., d_{n}*r)`.
    Where :math:`n` is the dimensionality of the data.
    See :class:`~torch.nn.PixelShuffle` for details.
    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by
    Examples::
        # 1D example
        #>>> input = torch.Tensor(1, 4, 8)
        #>>> output = F.pixel_shuffle(input, 2)
        #>>> print(output.size())
        torch.Size([1, 2, 16])
        # 2D example
        #>>> input = torch.Tensor(1, 9, 8, 8)
        #>>> output = F.pixel_shuffle(input, 3)
        #>>> print(output.size())
        torch.Size([1, 1, 24, 24])
        # 3D example
        #>>> input = torch.Tensor(1, 8, 16, 16, 16)
        #>>> output = F.pixel_shuffle(input, 2)
        #>>> print(output.size())
        torch.Size([1, 1, 32, 32, 32])
    """
    input_size = list(input.size())
    dimensionality = len(input_size) - 2

    input_size[1] //= (upscale_factor ** dimensionality)
    output_size = [dim * upscale_factor for dim in input_size[2:]]

    input_view = input.contiguous().view(
        input_size[0], input_size[1],
        *(([upscale_factor] * dimensionality) + input_size[2:])
    )

    indicies = list(range(2, 2 + 2 * dimensionality))
    indicies = indicies[1::2] + indicies[0::2]

    shuffle_out = input_view.permute(0, 1, *(indicies[::-1])).contiguous()
    return shuffle_out.view(input_size[0], input_size[1], *output_size)

class PixelShuffleGeneric(nn.Module):
    r"""Rearranges elements in a Tensor of shape :math:`(N, C, d_{1}, d_{2}, ..., d_{n})` to a
    tensor of shape :math:`(N, C/(r^n), d_{1}*r, d_{2}*r, ..., d_{n}*r)`.
    Where :math:`n` is the dimensionality of the data.

    This is useful for implementing efficient sub-pixel convolution
    with a stride of :math:`1/r`.

    Input Tensor must have at least 3 dimensions, e.g. :math:`(N, C, d_{1})` for 1D data,
    but Tensors with any number of dimensions after :math:`(N, C, ...)` (where N is mini-batch size,
    and C is channels) are supported.

    Look at the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details

    Args:
        upscale_factor (int): factor to increase spatial resolution by

    Shape:
        - Input: :math:`(N, C, d_{1}, d_{2}, ..., d_{n})`
        - Output: :math:`(N, C/(r^n), d_{1}*r, d_{2}*r, ..., d_{n}*r)`
        Where :math:`n` is the dimensionality of the data, e.g. :math:`n-1` for 1D audio,
        :math:`n=2` for 2D images, etc.

    Examples::

        # 1D example
        #>>> ps = nn.PixelShuffle(2)
        #>>> input = torch.Tensor(1, 4, 8)
        #>>> output = ps(input)
        #>>> print(output.size())
        torch.Size([1, 2, 16])

        # 2D example
        #>>> ps = nn.PixelShuffle(3)
        #>>> input = torch.Tensor(1, 9, 8, 8)
        #>>> output = ps(input)
        #>>> print(output.size())
        torch.Size([1, 1, 24, 24])

        # 3D example
        #>>> ps = nn.PixelShuffle(2)
        #>>> input = torch.Tensor(1, 8, 16, 16, 16)
        #>>> output = ps(input)
        #>>> print(output.size())
        torch.Size([1, 1, 32, 32, 32])

    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upscale_factor):
        super(PixelShuffleGeneric, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_shuffle_generic(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)
