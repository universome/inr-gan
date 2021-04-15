import torch
from torch import nn

from .conv import ModulatedConv2d, Upsample


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=(1, 3, 3, 1)):
        super().__init__()

        self.upsample = Upsample(blur_kernel) if upsample else None

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            if self.upsample is not None:
                skip = self.upsample(skip)
            out = out + skip

        return out
