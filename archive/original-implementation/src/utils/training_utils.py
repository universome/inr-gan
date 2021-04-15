from typing import List

import numpy as np
from scipy.special import gamma
from scipy.stats import truncnorm
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config


def construct_optimizer(model: nn.Module, optim_config: Config):
    name_to_cls = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'rms_prop': torch.optim.RMSprop
    }

    if optim_config.has('groups'):
        groups = [{'params': getattr(model, g).parameters(), **optim_config.groups.get(g)} for g in sorted(optim_config.groups.keys())]
    else:
        groups = [{'params': model.parameters()}]

    return name_to_cls[optim_config.type](groups, **optim_config.kwargs)


def construct_scheduler(optim, scheduler_config: Config):
    if scheduler_config.type == 'step_lr':
        return torch.optim.lr_scheduler.StepLR(optim, **scheduler_config.kwargs)
    else:
        raise NotImplementedError(f'Unknown scheduler: {scheduler_config.type}')


def get_weight_vector(module: nn.Module) -> Tensor:
    return torch.cat([p.view(-1) for p in module.parameters()])


def generate_coords(batch_size: int, img_size: int) -> Tensor:
    row = torch.arange(0, img_size).float() / img_size # [img_size]
    x_coords = row.view(1, -1).repeat(img_size, 1) # [img_size, img_size]
    y_coords = x_coords.t() # [img_size, img_size]

    coords = torch.stack([x_coords, y_coords], dim=2) # [img_size, img_size, 2]
    coords = coords.view(-1, 2) # [img_size ** 2, 2]
    coords = coords.t().view(1, 2, img_size ** 2).repeat(batch_size, 1, 1) # [batch_size, 2, n_coords]

    return coords


def generate_var_sized_coords(aspect_ratios: List[float], img_size: int) -> Tensor:
    """
    Generates variable-sized coordinates for images with padding.
    This is actually done by generating "normal" coordinates, but using
    a range beyond [0, 1] for a shorter side.
    Aspect ratio is assumed to be equal to w/h.
    The goal of this functino is two constrain the spacing
    """
    coords = generate_coords(len(aspect_ratios), img_size) # [batch_size, 2, img_size ** 2]
    scales = [([1.0 / ar, 1.0] if ar < 1.0 else [1.0, ar]) for ar in aspect_ratios] # [batch_size, 2]
    scales = torch.tensor(scales).unsqueeze(2) # [batch_size, 2, 1]
    coords_scaled = coords * scales

    return coords_scaled


def generate_random_resolution_coords(batch_size: int, img_size: int, scale: float=None, min_scale: float=None) -> Tensor:
    """
    Generating random input coordinate patches.
    It's used when we train on random image patches of different resolution
    """
    assert (int(scale is None) + int(min_scale is None)) == 1, "Either scale or min_scale should be specified."

    if scale is None:
        sizes = np.random.rand(batch_size) * (1 - min_scale) + min_scale # Changing the range: [0, 1] => [min_scale, 1]
        scale = min_scale
    else:
        sizes = np.ones(batch_size) * scale # [batch_size]

    x_offsets = np.random.rand(batch_size) * (1 - scale) # [batch_size]
    y_offsets = np.random.rand(batch_size) * (1 - scale) # [batch_size]

    # Unfortunately, torch.linspace cannot operate on a batch of inputs
    x_coords = torch.from_numpy(np.linspace(x_offsets, x_offsets + sizes, img_size, dtype=np.float32)) # [img_size, batch_size]
    y_coords = torch.from_numpy(np.linspace(y_offsets, y_offsets + sizes, img_size, dtype=np.float32)) # [img_size, batch_size]

    x_coords = x_coords.view(1, img_size, batch_size).repeat(1, img_size, 1) # [img_size, img_size, batch_size]
    y_coords = y_coords.view(img_size, 1, batch_size).repeat(1, img_size, 1) # [img_size, img_size, batch_size]

    x_coords = x_coords.view(img_size ** 2, batch_size) # [img_size ** 2, batch_size]
    y_coords = y_coords.view(img_size ** 2, batch_size) # [img_size ** 2, batch_size]

    coords = torch.stack([x_coords, y_coords], dim=0).permute(2, 0, 1) # [batch_size, 2, img_size ** 2]

    return coords


def sample_noise(dist: str, z_dim: int, batch_size: int, correction: Config=None) -> Tensor:
    assert dist in {'normal', 'uniform'}, f'Unknown latent distribution: {dist}'

    if dist == 'normal':
        if not correction is None and correction.enabled and correction.type == 'truncated':
            r = correction.kwargs.truncation_factor
            z = truncnorm.rvs(a=-r, b=r, size=(batch_size, z_dim))
            z = torch.from_numpy(z).float()
        else:
            z = torch.randn(batch_size, z_dim)

            if not correction is None and correction.enabled and correction.type == 'projected':
                # https://math.stackexchange.com/questions/827826/average-norm-of-a-n-dimensional-vector-given-by-a-normal-distribution
                norm = (1 / np.sqrt(2)) * z_dim * gamma((z_dim + 1)/2) / gamma((z_dim + 2)/2)
                # norm = np.sqrt(z_dim) # A fast approximation
                z /= z.norm(dim=1, keepdim=True)
                z *= norm
    elif dist == 'uniform':
        assert correction is None or correction.enabled == False, f"Unimplemented correction for uniform dist: {correction}"
        z = torch.rand(batch_size, z_dim) * 2 - 1

    return z


# def truncated_normal(size: torch.Size, mean: float=0.0, std: float=1.0):
#     result = torch.empty(size)

#     tmp = torch.randn(size + (4,))
#     valid = (tmp < 2) & (tmp > -2)
#     ind = valid.max(-1, keepdim=True)[1]

#     result = tmp.gather(-1, ind).squeeze(-1).view(size)
#     result.mul_(std).add_(mean)

#     return result

def compute_covariance(feats: Tensor) -> Tensor:
    """
    Computes empirical covariance matrix for a batch of feature vectors
    """
    assert feats.ndim == 2

    feats -= feats.mean(dim=0)
    cov_unscaled = feats.t() @ feats # [feat_dim, feat_dim]
    cov = cov_unscaled / (feats.size(0) - 1)

    return cov
