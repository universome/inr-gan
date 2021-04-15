from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig

from torch_utils import persistence
from torch_utils.ops import bias_act
from torch_utils import misc

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


#----------------------------------------------------------------------------

@persistence.persistent_class
class GenInput(nn.Module):
    def __init__(self, cfg: DictConfig, channel_dim: int, w_dim: int):
        super().__init__()

        self.cfg = cfg

        if self.cfg.type == 'const':
            self.input = torch.nn.Parameter(torch.randn([channel_dim, self.cfg.resolution, self.cfg.resolution]))
            self.total_dim = channel_dim
        elif self.cfg.type == 'coords':
            self.input = CoordsInput(self.cfg, w_dim)
            self.total_dim = self.input.get_total_dim()
        else:
            raise NotImplementedError

    def forward(self, batch_size: int, w: Tensor=None, device=None, dtype=None, memory_format=None) -> Tensor:
        if self.cfg.type == 'const':
            x = self.input.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([batch_size, 1, 1, 1])
        elif self.cfg.type == 'coords':
            x = self.input(batch_size, w, device=device, dtype=dtype, memory_format=memory_format)
        else:
            raise NotImplementedError

        return x


#----------------------------------------------------------------------------

@persistence.persistent_class
class CoordsInput(nn.Module):
    def __init__(self, cfg: DictConfig, w_dim: int):
        super().__init__()

        self.cfg = cfg
        self.coord_fuser = CoordFuser(self.cfg.coord_fuser_cfg, w_dim, self.cfg.resolution)

    def get_total_dim(self) -> int:
        return self.coord_fuser.total_dim

    def forward(self, batch_size: int, w: Optional[Tensor]=None, device='cpu', dtype=None, memory_format=None) -> Tensor:
        dummy_input = torch.empty(batch_size, 0, self.cfg.resolution, self.cfg.resolution)
        dummy_input = dummy_input.to(device, dtype=dtype, memory_format=memory_format)
        out = self.coord_fuser(dummy_input, w, dtype=dtype, memory_format=memory_format)

        return out


#----------------------------------------------------------------------------

@persistence.persistent_class
class CoordFuser(nn.Module):
    """
    CoordFuser which concatenates coordinates across dim=1 (we assume channel_first format)
    """
    def __init__(self, cfg: DictConfig, w_dim: int, resolution: int):
        super().__init__()

        self.cfg = cfg
        self.resolution = resolution
        self.res_cfg = self.cfg.res_configs[str(resolution)]
        self.log_emb_size = self.res_cfg.get('log_emb_size', 0)
        self.random_emb_size = self.res_cfg.get('random_emb_size', 0)
        self.shared_emb_size = self.res_cfg.get('shared_emb_size', 0)
        self.predictable_emb_size = self.res_cfg.get('predictable_emb_size', 0)
        self.const_emb_size = self.res_cfg.get('const_emb_size', 0)
        self.fourier_scale = self.res_cfg.get('fourier_scale', np.sqrt(10))
        self.use_cosine = self.res_cfg.get('use_cosine', False)
        self.use_raw_coords = self.res_cfg.get('use_raw_coords', False)
        self.init_dist = self.res_cfg.get('init_dist', 'randn')
        self._fourier_embs_cache = None
        self._full_cache = None
        self.use_full_cache = cfg.get('use_full_cache', False)

        if self.log_emb_size > 0:
            self.register_buffer('log_basis', generate_logarithmic_basis(
                resolution, self.log_emb_size, use_diagonal=self.res_cfg.get('use_diagonal', False))) # [log_emb_size, 2]

        if self.random_emb_size > 0:
            self.register_buffer('random_basis', self.sample_w_matrix((self.random_emb_size, 2), self.fourier_scale))

        if self.shared_emb_size > 0:
            self.shared_basis = nn.Parameter(self.sample_w_matrix((self.shared_emb_size, 2), self.fourier_scale))

        if self.predictable_emb_size > 0:
            self.W_size = self.predictable_emb_size * self.cfg.coord_dim
            self.b_size = self.predictable_emb_size
            self.affine = FullyConnectedLayer(w_dim, self.W_size + self.b_size, bias_init=0)

        if self.const_emb_size > 0:
            self.const_embs = nn.Parameter(torch.randn(1, self.const_emb_size, resolution, resolution).contiguous())

        self.total_dim = self.get_total_dim()
        self.is_modulated = (self.predictable_emb_size > 0)

    def sample_w_matrix(self, shape: Tuple[int], scale: float):
        if self.init_dist == 'randn':
            return torch.randn(shape) * scale
        elif self.init_dist == 'rand':
            return (torch.rand(shape) * 2 - 1) * scale
        else:
            raise NotImplementedError(f"Unknown init dist: {self.init_dist}")

    def get_total_dim(self) -> int:
        if self.cfg.fallback:
            return 0

        total_dim = 0
        total_dim += (self.cfg.coord_dim if self.use_raw_coords else 0)
        if self.log_emb_size > 0:
            total_dim += self.log_basis.shape[0] * (2 if self.use_cosine else 1)
        total_dim += self.random_emb_size * (2 if self.use_cosine else 1)
        total_dim += self.shared_emb_size * (2 if self.use_cosine else 1)
        total_dim += self.predictable_emb_size * (2 if self.use_cosine else 1)
        total_dim += self.const_emb_size

        return total_dim

    def forward(self, x: Tensor, w: Tensor=None, dtype=None, memory_format=None) -> Tensor:
        """
        Dims:
            @arg x is [batch_size, in_channels, img_size, img_size]
            @arg w is [batch_size, w_dim]
            @return out is [batch_size, in_channels + fourier_dim + cips_dim, img_size, img_size]
        """
        assert memory_format is torch.contiguous_format

        if self.cfg.fallback:
            return x

        batch_size, in_channels, img_size = x.shape[:3]
        out = x

        if self.use_full_cache and (not self._full_cache is None) and (self._full_cache.device == x.device) and \
           (self._full_cache.shape == (batch_size, self.get_total_dim(), img_size, img_size)):
           return torch.cat([x, self._full_cache], dim=1)

        if (not self._fourier_embs_cache is None) and (self._fourier_embs_cache.device == x.device) and \
           (self._fourier_embs_cache.shape == (batch_size, self.get_total_dim() - self.const_emb_size, img_size, img_size)):
            out = torch.cat([out, self._fourier_embs_cache], dim=1)
        else:
            raw_embs = []
            raw_coords = generate_coords(batch_size, img_size, x.device) # [batch_size, coord_dim, img_size, img_size]

            if self.use_raw_coords:
                out = torch.cat([out, raw_coords], dim=1)

            if self.log_emb_size > 0:
                log_bases = self.log_basis.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, log_emb_size, 2]
                raw_log_embs = torch.einsum('bdc,bcxy->bdxy', log_bases, raw_coords) # [batch_size, log_emb_size, img_size, img_size]
                raw_embs.append(raw_log_embs)

            if self.random_emb_size > 0:
                random_bases = self.random_basis.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, random_emb_size, 2]
                raw_random_embs = torch.einsum('bdc,bcxy->bdxy', random_bases, raw_coords) # [batch_size, random_emb_size, img_size, img_size]
                raw_embs.append(raw_random_embs)

            if self.shared_emb_size > 0:
                shared_bases = self.shared_basis.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, shared_emb_size, 2]
                raw_shared_embs = torch.einsum('bdc,bcxy->bdxy', shared_bases, raw_coords) # [batch_size, shared_emb_size, img_size, img_size]
                raw_embs.append(raw_shared_embs)

            if self.predictable_emb_size > 0:
                misc.assert_shape(w, [batch_size, None])
                mod = self.affine(w) # [batch_size, W_size + b_size]
                W = self.fourier_scale * mod[:, :self.W_size] # [batch_size, W_size]
                W = W.view(batch_size, self.predictable_emb_size, self.cfg.coord_dim) # [batch_size, predictable_emb_size, coord_dim]
                bias = mod[:, self.W_size:].view(batch_size, self.predictable_emb_size, 1, 1) # [batch_size, predictable_emb_size, 1]
                raw_predictable_embs = (torch.einsum('bdc,bcxy->bdxy', W, raw_coords) + bias) # [batch_size, predictable_emb_size, img_size, img_size]
                raw_embs.append(raw_predictable_embs)

            if len(raw_embs) > 0:
                raw_embs = torch.cat(raw_embs, dim=1) # [batch_suze, log_emb_size + random_emb_size + predictable_emb_size, img_size, img_size]
                raw_embs = raw_embs.contiguous() # [batch_suze, -1, img_size, img_size]
                out = torch.cat([out, raw_embs.sin().to(dtype=dtype, memory_format=memory_format)], dim=1) # [batch_size, -1, img_size, img_size]

                if self.use_cosine:
                    out = torch.cat([out, raw_embs.cos().to(dtype=dtype, memory_format=memory_format)], dim=1) # [batch_size, -1, img_size, img_size]

        if self.predictable_emb_size == 0 and self.shared_emb_size == 0 and out.shape[1] > x.shape[1]:
            self._fourier_embs_cache = out[:, x.shape[1]:].detach()

        if self.const_emb_size > 0:
            const_embs = self.const_embs.repeat([batch_size, 1, 1, 1])
            const_embs = const_embs.to(dtype=dtype, memory_format=memory_format)
            out = torch.cat([out, const_embs], dim=1) # [batch_size, total_dim, img_size, img_size]

        if self.use_full_cache and self.predictable_emb_size == 0 and self.shared_emb_size == 0 and out.shape[1] > x.shape[1]:
            self._full_cache = out[:, x.shape[1]:].detach()

        return out


def generate_coords(batch_size: int, img_size: int, device='cpu', align_corners: bool=False) -> Tensor:
    """
    Generates the coordinates in [-1, 1] range for a square image
    if size (img_size x img_size) in such a way that
    - upper left corner: coords[0, 0] = (-1, -1)
    - upper right corner: coords[img_size - 1, img_size - 1] = (1, 1)
    """
    if align_corners:
        row = torch.linspace(-1, 1, img_size, device=device).float() # [img_size]
    else:
        row = (torch.arange(0, img_size, device=device).float() / img_size) * 2 - 1 # [img_size]
    x_coords = row.view(1, -1).repeat(img_size, 1) # [img_size, img_size]
    y_coords = x_coords.t().flip(dims=(0,)) # [img_size, img_size]

    coords = torch.stack([x_coords, y_coords], dim=2) # [img_size, img_size, 2]
    coords = coords.view(-1, 2) # [img_size ** 2, 2]
    coords = coords.t().view(1, 2, img_size, img_size).repeat(batch_size, 1, 1, 1) # [batch_size, 2, img_size, img_size]

    return coords


def generate_logarithmic_basis(
    resolution: int,
    max_num_feats: int=np.float('inf'),
    remove_lowest_freq: bool=False,
    use_diagonal: bool=True) -> Tensor:
    """
    Generates a directional logarithmic basis with the following directions:
        - horizontal
        - vertical
        - main diagonal
        - anti-diagonal
    """
    max_num_feats_per_direction = np.ceil(np.log2(resolution)).astype(int)
    bases = [
        generate_horizontal_basis(max_num_feats_per_direction),
        generate_vertical_basis(max_num_feats_per_direction),
    ]

    if use_diagonal:
        bases.extend([
            generate_diag_main_basis(max_num_feats_per_direction),
            generate_anti_diag_basis(max_num_feats_per_direction),
        ])

    if remove_lowest_freq:
        bases = [b[1:] for b in bases]

    # If we do not fit into `max_num_feats`, then trying to remove the features in the order:
    # 1) anti-diagonal 2) main-diagonal
    # while (max_num_feats_per_direction * len(bases) > max_num_feats) and (len(bases) > 2):
    #     bases = bases[:-1]

    basis = torch.cat(bases, dim=0)

    # If we still do not fit, then let's remove each second feature,
    # then each third, each forth and so on
    # We cannot drop the whole horizontal or vertical direction since otherwise
    # model won't be able to locate the position
    # (unless the previously computed embeddings encode the position)
    # while basis.shape[0] > max_num_feats:
    #     num_exceeding_feats = basis.shape[0] - max_num_feats
    #     basis = basis[::2]

    assert basis.shape[0] <= max_num_feats, \
        f"num_coord_feats > max_num_fixed_coord_feats: {basis.shape, max_num_feats}."

    return basis


def generate_horizontal_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [0.0, 1.0], 4.0)


def generate_vertical_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [1.0, 0.0], 4.0)


def generate_diag_main_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))


def generate_anti_diag_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))


def generate_wavefront_basis(num_feats: int, basis_block: List[float], period_length: float) -> Tensor:
    period_coef = 2.0 * np.pi / period_length
    basis = torch.tensor([basis_block]).repeat(num_feats, 1) # [num_feats, 2]
    powers = torch.tensor([2]).repeat(num_feats).pow(torch.arange(num_feats)).unsqueeze(1) # [num_feats, 1]
    result = basis * powers * period_coef # [num_feats, 2]

    return result.float()
