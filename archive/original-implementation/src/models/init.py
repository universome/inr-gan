"""
Copypasted from https://github.com/dalmia/siren
"""
import numpy as np
import torch
from torch.nn.init import _calculate_correct_fan


def siren_uniform_(tensor: torch.Tensor, mode: str = 'fan_in', c: float = 6):
    r"""Fills the input `Tensor` with values according to the method
    described in ` Implicit Neural Representations with Periodic Activation
    Functions.` - Sitzmann, Martel et al. (2020), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \sqrt{\frac{6}{\text{fan\_mode}}}
    Also known as Siren initialization.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> siren.init.siren_uniform_(w, mode='fan_in', c=6)
    :param tensor: an n-dimensional `torch.Tensor`
    :type tensor: torch.Tensor
    :param mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing
        ``'fan_in'`` preserves the magnitude of the variance of the weights in
        the forward pass. Choosing ``'fan_out'`` preserves the magnitudes in
        the backwards pass.s
    :type mode: str, optional
    :param c: value used to compute the bound. defaults to 6
    :type c: float, optional
    """
    fan = _calculate_correct_fan(tensor, mode)
    std = 1 / np.sqrt(fan)
    bound = np.sqrt(c) * std  # Calculate uniform bounds from standard deviation

    with torch.no_grad():
        # In this case we have std of bound / sqrt(3)
        return tensor.uniform_(-bound, bound)


def compute_dense_scale(in_features: int) -> float:
    # Using Kaiming normal scale for relu
    return np.sqrt(2 / in_features)


def compute_siren_std(in_features: int, is_coord_layer: bool) -> float:
    if is_coord_layer:
        # https://github.com/vsitzmann/siren/blob/ecd150f99b40217d76e0f15753b856aa2d966ab1/modules.py#L629-L634
        bound = 1 / in_features
    else:
        # https://github.com/vsitzmann/siren/blob/ecd150f99b40217d76e0f15753b856aa2d966ab1/modules.py#L621-L626
        bound = np.sqrt(6 / in_features) / 30

    return bound / np.sqrt(3)


def compute_siren_std_without_w0(in_features: int) -> float:
    return np.sqrt(2 / in_features)


def compute_old_siren_std(in_features: int, w0: float) -> float:
    """
    Official implementation does not work that well for us :|
    """
    bound = np.sqrt(6 / in_features) / w0
    std = bound / np.sqrt(3)

    return std
