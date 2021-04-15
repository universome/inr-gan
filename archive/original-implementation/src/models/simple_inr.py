import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.models.layers import create_activation


class SimpleINR(nn.Module):
    """
    Basic INR model (with ReLU activaions) that is used as a baseline
    """
    def __init__(self, config: Config):
        super(SimpleINR, self).__init__()

        layers = [
            nn.Linear(config.in_features, config.layer_sizes[0], bias=config.has_bias),
            create_activation(config.activation)
        ]

        for index in range(len(config.layer_sizes) - 1):
            layers.extend([
                nn.Linear(config.layer_sizes[index], config.layer_sizes[index + 1], bias=config.has_bias),
                create_activation(config.activation)
            ])

        layers.extend([
            nn.Linear(config.layer_sizes[-1], config.out_features, bias=config.has_bias),
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, coords: Tensor) -> Tensor:
        return self.model(coords)
