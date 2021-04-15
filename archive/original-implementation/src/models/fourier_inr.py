import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.models.layers import create_activation, LinearResidual


class FourierINR(nn.Module):
    """
    INR with Fourier features as specified in https://people.eecs.berkeley.edu/~bmild/fourfeat/
    """
    def __init__(self, config: Config):
        super(FourierINR, self).__init__()

        layers = [
            nn.Linear(config.num_fourier_feats * 2, config.layer_sizes[0], bias=config.has_bias),
            create_activation(config.activation)
        ]

        for index in range(len(config.layer_sizes) - 1):
            transform = nn.Sequential(
                nn.Linear(config.layer_sizes[index], config.layer_sizes[index + 1], bias=config.has_bias),
                create_activation(config.activation)
            )

            if config.residual.enabled:
                layers.append(LinearResidual(config.residual, transform))
            else:
                layers.append(transform)

        layers.append(nn.Linear(config.layer_sizes[-1], config.out_features, bias=config.has_bias))

        self.model = nn.Sequential(*layers)

        # Initializing the basis
        basis_matrix = config.scale * torch.randn(config.num_fourier_feats, config.in_features)
        self.basis_matrix = nn.Parameter(basis_matrix, requires_grad=config.learnable_basis)

    def compute_fourier_feats(self, coords: Tensor) -> Tensor:
        sines = (2 * np.pi * coords @ self.basis_matrix.t()).sin() # [batch_size, num_fourier_feats]
        cosines = (2 * np.pi * coords @ self.basis_matrix.t()).cos() # [batch_size, num_fourier_feats]

        return torch.cat([sines, cosines], dim=1) # [batch_size, num_fourier_feats * 2]

    def forward(self, coords: Tensor) -> Tensor:
        return self.model(self.compute_fourier_feats(coords))
