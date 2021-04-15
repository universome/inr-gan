import math
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..op import fused_leaky_relu, FusedLeakyReLU


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"


class StyledLinear(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            style_dim,
            modulation_type: str='fmm_inrgan',
            factorization_rank: Optional[int] = None):
        super().__init__()

        if modulation_type in ['fmm_inrgan', 'fmm_sg2']:
            self.transform = FMMLinear(in_channel, out_channel, style_dim, modulation_type, factorization_rank)
        elif modulation_type == 'sg2':
            self.transform = ModulatedLinear(in_channel, out_channel, style_dim)
        else:
            raise NotImplementedError(f'Unknown modulation type: {modulation_type}')

        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.transform(input, style)
        out = self.activate(out)

        return out


class ModulatedLinear(nn.Module):
    """
    Demod Linear layer via BMM instead of group conv
    """
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        style_dim: int):

        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weights_scale = (1.0 / math.sqrt(in_channel))
        self.style_to_mod = EqualLinear(style_dim, in_channel, bias_init=1.0)
        self.shared_weights = nn.Parameter(torch.randn(1, out_channel, in_channel))

    def forward(self, input: Tensor, style: Tensor) -> Tensor:
        """
        input: [batch_size, dim, height, width]
        style: [batch_size, style_dim]
        """
        batch_size, in_channel, height, width = input.shape

        # Computing the weight
        modulation = self.style_to_mod(style).view(batch_size, 1, self.in_channel) # [batch_size, 1, in_channel]
        weights = self.weights_scale * self.shared_weights * modulation # [batch_size, out_channel, in_channel]
        weights = weights / (weights.norm(dim=2, keepdim=True) + 1e-8) # [batch_size, out_channel, in_channel]

        # Computing the out
        input = input.view(batch_size, in_channel, height * width) # [batch_size, in_channel, height * width]
        out = weights @ input # [batch_size, out_channel, height * width]
        out = out.view(batch_size, self.out_channel, height, width) # [batch_size, out_channel, height, width]

        return out


class FMMLinear(nn.Module):
    """
    FMM layer via BMM instead of F.conv
    """
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        style_dim: int,
        modulation_type: str,
        factorization_rank: int):

        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.rank = factorization_rank
        self.weights_scale = (1.0 / math.sqrt(in_channel))
        self.num_external_params = out_channel * self.rank + in_channel * self.rank
        # bias_init is 0 and not 1 since FMM modulation is different from AdaIN
        self.style_to_mod = EqualLinear(style_dim, self.num_external_params, bias_init=0.0)
        self.shared_weights = nn.Parameter(torch.randn(out_channel, in_channel))
        self.modulation_type = modulation_type

    def forward(self, input: Tensor, style: Tensor) -> Tensor:
        """
        input: [batch_size, dim, height, width]
        style: [batch_size, style_dim]
        """
        batch_size, in_channel, height, width = input.shape

        # Now, we need to construct a [out_channel, in_channel] matrix
        params = self.style_to_mod(style) # [batch_size, num_params]
        left_matrix = params[:, :self.out_channel * self.rank] # [batch_size, left_matrix_size]
        right_matrix = params[:, self.out_channel * self.rank:] # [batch_size, right_matrix_size]

        left_matrix = left_matrix.view(batch_size, self.out_channel, self.rank) # [batch_size, out_channel, rank]
        right_matrix = right_matrix.view(batch_size, self.rank, self.in_channel) # [batch_size, rank, in_channel]

        # Imagine, that the output of `self.style_to_params` is N(0, 1)
        # Then, std of weights is sqrt(self.rank). Converting it back to N(0, 1)
        modulation = left_matrix @ right_matrix # [batch_size, out_channel, in_channel]
        modulation /= np.sqrt(self.rank) # [batch_size, out_channel, in_channel]

        if self.modulation_type == "fmm_inrgan":
            modulation = modulation.tanh() + 1.0 # [batch_size, out_channel, in_channel]
            weights = self.shared_weights.unsqueeze(0) * self.weights_scale * modulation # [batch_size, out_channel, in_channel]
        elif self.modulation_type == "fmm_sg2":
            # Let's modulate as similar as possible to stylegan2 modulation, but just with a higher rank
            weights = self.shared_weights.unsqueeze(0) * self.weights_scale * (modulation + 1.0) # [batch_size, out_channel, in_channel]
            weights = weights / (weights.norm(dim=2, keepdim=True) + 1e-8) # [batch_size, out_channel, in_channel]
        else:
            raise NotImplementedError

        input = input.view(batch_size, in_channel, height * width) # [batch_size, in_channel, height * width]
        out = weights @ input # [batch_size, out_channel, height * width]
        out = out.view(batch_size, self.out_channel, height, width) # [batch_size, out_channel, height, width]

        return out
