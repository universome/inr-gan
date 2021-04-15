"""
LSUN bedroom model from https://github.com/LMescheder/GAN_stability
(https://arxiv.org/abs/1801.04406)
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import numpy as np
from firelab.config import Config


class ResnetGAN(nn.Module):
    def __init__(self, config: Config):
        super(ResnetGAN, self).__init__()

        self.generator = ResnetGenerator(config)
        self.discriminator = ResnetDiscriminator(config)


class ResnetGenerator(nn.Module):
    # def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, nfilter_max=512, **kwargs):
    def __init__(self, config: Config):
        super(ResnetGenerator, self).__init__()

        self.config = config.hp.generator

        # Submodules
        nlayers = int(np.log2(config.data.target_img_size / self.config.s0))
        self.nf0 = min(self.config.nfilter_max, self.config.nfilter * 2 ** nlayers)
        self.fc = nn.Linear(self.config.z_dim, self.nf0 * (self.config.s0 ** 2))

        blocks = []
        for i in range(nlayers):
            nf0 = min(self.config.nfilter * 2 ** (nlayers-i), self.config.nfilter_max)
            nf1 = min(self.config.nfilter * 2 ** (nlayers-i-1), self.config.nfilter_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [ResnetBlock(self.config.nfilter, self.config.nfilter)]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(self.config.nfilter, 3, 3, padding=1)

    def sample_noise(self, batch_size: int):
        return torch.randn(batch_size, self.config.z_dim)

    def generate_image(self, batch_size: int, device: str) -> Tensor:
        return self.forward(self.sample_noise(batch_size).to(device))

    def forward(self, z: Tensor, *args, **kwargs) -> Tensor:
        batch_size = z.size(0)
        out = self.fc(z)
        out = out.view(batch_size, self.nf0, self.config.s0, self.config.s0)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))
        out = torch.tanh(out)

        return out


class ResnetDiscriminator(nn.Module):
    # def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, nfilter_max=1024):
    def __init__(self, config: Config):
        super(ResnetDiscriminator, self).__init__()

        self.config = config.hp.discriminator
        size = config.data.target_img_size
        nf = self.config.nfilter
        nf_max = self.config.nfilter_max

        # Submodules
        nlayers = int(np.log2(size / self.config.s0))
        self.nf0 = min(self.config.nfilter_max, nf * 2 ** nlayers)

        blocks = [ResnetBlock(nf, nf)]

        for i in range(nlayers):
            nf0 = min(nf * 2 ** i, self.config.nfilter_max)
            nf1 = min(nf * 2 ** (i+1), self.config.nfilter_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * self.config.s0 * self.config.s0, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(x.size(0), self.nf0 * (self.config.s0 ** 2))
        out = self.fc(actvn(out))

        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1 * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out
