from functools import reduce
from operator import mul
from typing import Union, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FourierMapping(nn.Module):
    """
    Fourier features as specified in https://people.eecs.berkeley.edu/~bmild/fourfeat/
    with additional custom modifications
    """
    def __init__(self, in_features: int, out_features: int,
                 scale: Union[float, List[float]] = 1.0, learnable: bool = False,
                 init: str = 'log',
                 compute_sin: bool = True, compute_cos: bool = True):
        super().__init__()

        assert compute_sin or compute_cos
        assert scale > 0

        if compute_sin and compute_cos:
            assert out_features % 2 == 0

        self.in_features = in_features
        self.out_features = out_features
        self.compute_sin = compute_sin
        self.compute_cos = compute_cos
        self.init = init

        out_size = (out_features // 2) if (self.compute_sin and self.compute_cos) else out_features

        if init == 'standard':
            basis = scale * torch.randn(out_size, in_features)
        elif init == 'uniform':
            basis = torch.FloatTensor(out_size, in_features).uniform_(-scale, scale)
        elif init == 'log':
            if isinstance(scale, int) or isinstance(scale, float):
                scale = [scale for k in range(in_features)]
            else:
                assert len(scale) == in_features

            basis = []
            for i in range(in_features):
                basis.append(torch.logspace(start=-3, end=scale[i], base=2., steps=out_size - 2) * np.pi)
                basis[-1] = torch.cat((basis[-1], torch.tensor([0.0, 0.1])))
            basis = torch.stack(basis, dim=1)
        else:
            raise NotImplementedError(f'Unknown init: {init}')

        basis = basis.t() # (C_in, C_out)

        if learnable:
            self.register_parameter("basis", nn.Parameter(basis))
        else:
            self.register_buffer("basis", basis)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): of shape (B, C_in)
                e.g. in case of images C_in = 2 (x and y coordinates)
        Returns:
            Tensor: of shape (B, C_out)
        """
        if self.init == 'standard':
            weight = (2.0 * np.pi) * self.basis
        else:
            weight = self.basis

        out = x @ weight  # (B, C_out) or (B, C_out // 2) if both sin and cos

        if self.compute_sin and self.compute_cos:
            return torch.cat([out.sin(), out.cos()], dim=1)
        elif self.compute_sin:
            return out.sin()
        else:
            return out.cos()


class CoordinateEmbedding(nn.Module):
    """
    Coordinate embeddings as specified in https://arxiv.org/abs/2011.13775
    """
    def __init__(self, embedding_dim: int, size: Union[Tuple[int, int], Tuple[int, int, int]]):
        super().__init__()
        self.embedding_dim = embedding_dim
        num_pixels = float(reduce(mul, size))
        self.embeddings = nn.Parameter(
            torch.FloatTensor(1, embedding_dim, *size).uniform_(-1.0 / num_pixels, 1.0 / num_pixels))

    def forward(self, grid: torch.Tensor, return_embeddings=False):
        """
        Args:
            grid (Tensor): of shape (B, H_out, W_out, 2) or (B, D_out, H_out, W_out, 3)
        Returns:
            Tensor: of shape (B, embedding_dim, H_out, W_out) or (B, embedding_dim, D_out, H_out, W_out)
        """
        if return_embeddings:
            return self.embeddings.expand(grid.shape[0], *self.embeddings.shape[1:]) \
                .view(grid.shape[0], self.embedding_dim, *grid.shape[1:-1])
        else:
            return F.grid_sample(
                self.embeddings.expand(grid.shape[0], *self.embeddings.shape[1:]), grid, align_corners=True)