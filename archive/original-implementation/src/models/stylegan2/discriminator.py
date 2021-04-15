"""
Copy-pasted from https://github.com/rosinality/stylegan2-pytorch
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from firelab.config import Config

from src.models.layers import EqualLinear, EqualConv2d, ScaledLeakyReLU, make_kernel
from .op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        return upfirdn2d(input, self.kernel, pad=self.pad)


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * config.hp.discriminator.channel_multiplier,
            128: 128 * config.hp.discriminator.channel_multiplier,
            256: 64 * config.hp.discriminator.channel_multiplier,
            512: 32 * config.hp.discriminator.channel_multiplier,
            1024: 16 * config.hp.discriminator.channel_multiplier,
        }

        convs = [ConvLayer(3, channels[config.data.target_img_size], 1)]
        log_size = int(math.log(config.data.target_img_size, 2))
        in_channel = channels[config.data.target_img_size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, config.hp.discriminator.blur_kernel))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        if self.config.data.name == 'imagenet_vs':
            num_output_logits = self.config.data.num_classes
        else:
            num_output_logits = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], num_output_logits),
        )

    def forward(self, input, labels=None):
        assert labels is None or len(input) == len(labels), f"Wrong lens: {input.shape} vs {labels.shape}"
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(batch, -1)
        out = self.final_linear(out)

        if labels is None:
            return out
        else:
            return out[range(len(labels)), labels]
