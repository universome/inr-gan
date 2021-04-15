from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from firelab.config import Config

from src.models.layers import (
    Reshape,
    ResidualBlock,
    OptimizedResidualBlock,
    sn_wrapper,
    EqualLinear,
    EqualConv2d,
    ScaledLeakyReLU,
)


class GAN32(nn.Module):
    def __init__(self, config: Config):
        super(GAN32, self).__init__()
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)


class Generator(nn.Module):
    def __init__(self, config: Config):
        # latent_dim=128, img_size=32, image_channels=3, channels=128, main_branch_weight=0.1
        super(Generator, self).__init__()

        self.config = config
        z_dim = self.config.hp.generator.z_dim
        dim = self.config.hp.generator.dim
        main_branch_weight = self.config.hp.generator.main_branch_weight
        img_size = self.config.data.target_img_size

        self.model = nn.Sequential(
            EqualLinear(z_dim, dim * 4 * 4),
            Reshape([-1, dim, 4, 4]),
            ResidualBlock(dim, dim // 2, 3, resample='up', main_branch_weight=main_branch_weight),
            ResidualBlock(dim // 2, dim // 4, 3, resample='up', main_branch_weight=main_branch_weight),
            ResidualBlock(dim // 4, dim // 8, 3, resample='up', main_branch_weight=main_branch_weight),
            ResidualBlock(dim // 8, dim // 16, 3, resample='up', main_branch_weight=main_branch_weight),
            nn.ReLU(),
            EqualConv2d(dim // 16, self.config.data.num_img_channels, 3, padding=1),
            nn.Tanh(),
        )

    def sample_noise(self, batch_size: int):
        return torch.randn(batch_size, self.config.hp.generator.z_dim)

    def generate_image(self, batch_size: int, device: str) -> Tensor:
        return self.forward(self.sample_noise(batch_size).to(device))

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, config: Config):
        super(Discriminator, self).__init__()

        self.config = config
        dim = self.config.hp.discriminator.dim
        main_branch_weight = self.config.hp.discriminator.main_branch_weight
        use_sn = self.config.hp.discriminator.get('use_spectral_norm', False)

        num_input_channels = self.config.data.num_img_channels
        if self.config.data.get('concat_patches.enabled') and self.config.data.concat_patches.axis == 'channel':
            num_input_channels *= 2

        if self.config.hp.discriminator.n_layers == 4:
            layers = [
                ResidualBlock(dim, dim * 2, 3, resample='down', main_branch_weight=main_branch_weight, use_sn=use_sn),
                ResidualBlock(dim * 2, dim * 4, 3, resample='down', main_branch_weight=main_branch_weight, use_sn=use_sn),
                ResidualBlock(dim * 4, dim * 8, 3, resample='down', main_branch_weight=main_branch_weight, use_sn=use_sn),
                ResidualBlock(dim * 8, dim * 16, 3, resample='down', main_branch_weight=main_branch_weight, use_sn=use_sn),
            ]
        elif self.config.hp.discriminator.n_layers == 6:
            layers = [
                ResidualBlock(dim, dim * 2, 3, resample='down', main_branch_weight=main_branch_weight, use_sn=use_sn),
                ResidualBlock(dim * 2, dim * 4, 3, resample='down', main_branch_weight=main_branch_weight, use_sn=use_sn),
                ResidualBlock(dim * 4, dim * 4, 3, resample='down', main_branch_weight=main_branch_weight, use_sn=use_sn),
                ResidualBlock(dim * 4, dim * 8, 3, resample='down', main_branch_weight=main_branch_weight, use_sn=use_sn),
                ResidualBlock(dim * 8, dim * 8, 3, resample='down', main_branch_weight=main_branch_weight, use_sn=use_sn),
                ResidualBlock(dim * 8, dim * 16, 3, resample='down', main_branch_weight=main_branch_weight, use_sn=use_sn),
            ]
        else:
            raise NotImplementedError(f'Bad number of layers: {self.config.hp.discriminator.n_layers}')

        self.model = nn.Sequential(
            OptimizedResidualBlock(num_input_channels, dim, 3, main_branch_weight=main_branch_weight, use_sn=use_sn),
            *layers,
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            sn_wrapper(EqualLinear(dim * 16, 1), use_sn),
        )

    def forward(self, x: Tensor, labels=None) -> Tensor:
        assert labels is None
        return self.model(x)


class Discriminator32(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.model = nn.Sequential(
            Reshape([-1, config.data.num_img_channels, config.data.target_img_size, config.data.target_img_size]),
            EqualConv2d(config.data.num_img_channels, config.hp.discriminator.dim, kernel_size=5, stride=2, padding=2),
            ScaledLeakyReLU(),

            EqualConv2d(config.hp.discriminator.dim, 2 * config.hp.discriminator.dim, kernel_size=5, stride=2, padding=2),
            ScaledLeakyReLU(),

            EqualConv2d(2 * config.hp.discriminator.dim, 4 * config.hp.discriminator.dim, kernel_size=5, stride=2, padding=2),
            ScaledLeakyReLU(),

            nn.Flatten(),
            EqualLinear(4 * 4 * 4 * config.hp.discriminator.dim, 1)
        )

    def forward(self, x: Tensor):
        return self.model(x)
