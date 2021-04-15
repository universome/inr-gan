from typing import List

import numpy as np
import torch
from torch import nn
from torch import Tensor

from .linear import EqualLinear


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class CoordBasedConstantInput(nn.Module):
    def __init__(self, dim, size, use_coordinates_input: bool=False, modulatable: bool=False):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, dim, size, size))

        if modulatable:
            assert False
        else:
            self.style_to_params = None

        if use_coordinates_input:
            self.coords_input = CoordinatesInput(dim, size)
        else:
            self.coords_input = None

    def get_total_dim(self) -> int:
        if self.coords_input is None:
            return self.input.shape[1]
        else:
            return self.input.shape[1] + self.coords_input.dim

    def forward(self, input, image_size: int=None):
        batch_size = input.shape[0]
        const_out = self.input.repeat(batch_size, 1, 1, 1) # [batch_size, dim, img_size, img_size]

        if self.coords_input is None:
            out = const_out
        else:
            coords_out = self.coords_input(batch_size, const_out.device, image_size) # [batch_size, dim, img_size, img_size]
            out = torch.cat([const_out, coords_out], dim=1) # [batch_size, 2 * dim, img_size, img_size]

        return out



class CoordinatesInput(nn.Module):
    def __init__(self, dim: int, img_size):
        super().__init__()

        self.transform = nn.Linear(2, dim)
        self.img_size = img_size
        self.dim = dim

        nn.init.uniform_(self.transform.weight, -np.sqrt(9 / dim), np.sqrt(9 / dim))
        self.transform.bias.data.zero_()

    def forward(self, batch_size, device, img_size: int=None):
        img_size = img_size or self.img_size
        coords = generate_coords(batch_size, img_size, device) # [batch_size, 2, n_coords]
        coords = coords.permute(0, 2, 1) # [batch_size, n_coords, 2]
        embs = self.transform(coords).sin() # [batch_size, n_coords, dim]
        embs = embs.permute(0, 2, 1) # [batch_size, dim, n_coords]
        embs = embs.view(embs.shape[0], embs.shape[1], img_size, img_size) # [batch_size, dim, img_size, img_size]

        return embs


def generate_coords(batch_size: int, img_size: int, device='cpu') -> Tensor:
    # Generating the coordinates for a single row in [-1, 1] range
    row = (torch.arange(0, img_size, device=device).float() / img_size) * 2 - 1 # [img_size]
    x_coords = row.view(1, -1).repeat(img_size, 1) # [img_size, img_size]
    y_coords = x_coords.t() # [img_size, img_size]

    coords = torch.stack([x_coords, y_coords], dim=2) # [img_size, img_size, 2]
    coords = coords.view(-1, 2) # [img_size ** 2, 2]
    coords = coords.t().view(1, 2, img_size ** 2).repeat(batch_size, 1, 1) # [batch_size, 2, n_coords]

    return coords


class CoordFuser(nn.Module):
    """
    What this class does is that it takes an input and concatenates coordinates to it
    The coordinates are predictable from style
    """
    def __init__(self,
            emb_dim: int,
            style_dim: int,
            coord_dim: int,
            fourier_scale: float,
            fuse_type: str="concat",
            multi_scale_coord_embs: bool=False,
            img_size: int=None,
            use_log_coords: bool=False,
            no_fourier_embs: bool=False,
        ):
        """
        Some args:
            - multi_scale_coord_embs — if we should learn coord embeddings for each resolution like in CIPS for the input one
            - img_size — if multi_scale_coord_embs == True, then we should know the resolution
            - use_log_coords — if True, then one part of our basis is fixed
            - no_fourier_embs — you can save computation by NOT using fourier embeddings
        """

        super().__init__()

        assert coord_dim == 2, "Works for 2D images only for now."

        self.emb_dim = emb_dim
        self.coord_dim = coord_dim
        self.fourier_scale = fourier_scale
        self.fuse_type = fuse_type
        self.no_fourier_embs = no_fourier_embs

        if no_fourier_embs:
            pass
        else:
            self.W_size = coord_dim * emb_dim
            self.b_size = emb_dim
            self.style_to_mapping_params = EqualLinear(style_dim, self.W_size + self.b_size, bias_init=0.0)

            if use_log_coords:
                self.register_buffer('fixed_basis', generate_fixed_basis(img_size, emb_dim))
            else:
                self.fixed_basis = None

            if multi_scale_coord_embs:
                self.coord_embs = nn.Parameter(torch.randn(1, emb_dim, img_size, img_size))
            else:
                self.coord_embs = None

    def forward(self, input: Tensor, style: Tensor) -> Tensor:
        """
        Dims:
            @arg input is [batch_size, in_channels, img_size, img_size]
            @arg style is [batch_size, style_dim]
            @return out is [batch_size, in_channels + emb_dim, img_size, img_size]
        """
        batch_size, in_channels, img_size = input.shape[:3]
        coords = generate_coords(batch_size, img_size, input.device) # [batch_size, coord_dim, img_size ** 2]

        if self.no_fourier_embs:
            raw_coords = coords.view(batch_size, 2, img_size, img_size)
            return torch.cat([input, raw_coords], dim=1)

        mod = self.style_to_mapping_params(style) # [batch_size, W_size + b_size]
        W = self.fourier_scale * mod[:, :self.W_size].view(batch_size, self.emb_dim, self.coord_dim) # [batch_size, emb_dim, coord_dim]
        bias = mod[:, self.W_size:].unsqueeze(2) # [batch_size, emb_dim, 1]

        coord_embs = (W @ coords + bias).sin() # [batch_size, coord_dim, img_size ** 2]
        coord_embs = coord_embs.view(batch_size, self.emb_dim, img_size, img_size) # [batch_size, coord_dim, img_size, img_size]

        if self.fuse_type == "concat":
            out = torch.cat([input, coord_embs], dim=1) # [batch_size, in_channels + emb_dim, img_size, img_size]

            if not self.coord_embs is None:
                out = torch.cat([out, self.coord_embs.repeat(batch_size, 1, 1, 1)], dim=1) # [batch_size, in_channels + 2 * emb_dim, img_size, img_size]

            if not self.fixed_basis is None:
                # fixed_activations = F.linear(coords, self.fixed_basis) # [batch_size, emb_dim, img_size ** 2]
                fixed_basis = self.fixed_basis.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, num_fixed_feats, coord_dim]
                fixed_activations = fixed_basis @ coords # [batch_size, num_fixed_feats, img_size ** 2]
                fixed_activations = fixed_activations.view(batch_size, self.fixed_basis.shape[0], img_size, img_size) # [batch_size, num_fixed_feats, img_size ** 2]
                fixed_coord_embs = torch.cat([fixed_activations.sin(), fixed_activations.cos()], dim=1) # [batch_size, num_fixed_feats * 2, img_size, img_size]

                out = torch.cat([out, fixed_coord_embs])
        elif self.fuse_type == "add":
            assert in_channels == self.emb_dim
            out = input + coord_embs # [batch_size, in_channels, img_size, img_size]

            if not self.coord_embs is None:
                out = out + self.coord_embs # [batch_size, in_channels, img_size, img_size]

            assert self.fixed_basis is None
        elif self.fuse_type == "mult":
            assert in_channels == self.emb_dim
            out = input * coord_embs # [batch_size, in_channels, img_size, img_size]

            if not self.coord_embs is None:
                out = out * self.coord_embs # [batch_size, in_channels, img_size, img_size]

            assert self.fixed_basis is None
        else:
            raise NotImplementedError(f"Unknown fuse type: {self.fuse_type}")

        return out

    def get_total_dim(self) -> int:
        if self.no_fourier_embs:
            return self.coord_dim

        if self.fuse_type in ["add", "mult"]:
            return 0
        else:
            if self.coord_embs is None:
                return self.emb_dim
            else:
                return self.emb_dim + self.coord_embs.shape[1]


def generate_fixed_basis(resolution: int, max_num_feats: int, use_diag_feats: bool=False, select_high_freq_first: bool=True) -> Tensor:
    num_feats_per_direction = np.ceil(np.log2(resolution)).astype(int) + 1
    bases = [
        generate_horizontal_basis(num_feats_per_direction),
        generate_vertical_basis(num_feats_per_direction),
        generate_diag_main_basis(num_feats_per_direction),
        generate_anti_diag_basis(num_feats_per_direction),
    ]

    # First, trying to remove diag features
    if num_feats_per_direction * len(bases) > max_num_feats:
        if not use_diag_feats:
            bases = bases[:2]

    # Then, removing extra features...
    if num_feats_per_direction * len(bases) > max_num_feats:
        num_exceeding_feats = (num_feats_per_direction * len(bases) - max_num_feats) // len(bases)

        if select_high_freq_first:
            bases = [b[num_exceeding_feats:] for b in bases]
        else:
            bases = [b[:-num_exceeding_feats] for b in bases]

    basis = torch.cat(bases, dim=0)

    assert basis.shape[0] <= max_num_feats, \
        f"num_coord_feats > max_num_fixed_coord_feats: {basis.shape, max_num_feats}"

    return basis


def generate_horizontal_basis(num_feats: int) -> Tensor:
    return generate_basis(num_feats, [0.0, 1.0], 4.0)


def generate_vertical_basis(num_feats: int) -> Tensor:
    return generate_basis(num_feats, [1.0, 0.0], 4.0)


def generate_diag_main_basis(num_feats: int) -> Tensor:
    return generate_basis(num_feats, [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))


def generate_anti_diag_basis(num_feats: int) -> Tensor:
    return generate_basis(num_feats, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))


def generate_basis(num_feats: int, basis_block: List[float], period_length: float) -> Tensor:
    period_coef = 2.0 * np.pi / period_length
    basis = torch.tensor([basis_block]).repeat(num_feats, 1) # [num_feats, 2]
    powers = torch.tensor([2]).repeat(num_feats).pow(torch.arange(num_feats)).unsqueeze(1) # [num_feats, 1]
    result = basis * powers * period_coef # [num_feats, 2]

    return result.float()
