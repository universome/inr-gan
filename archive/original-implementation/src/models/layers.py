import math
import random
from typing import Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from firelab.config import Config

from src.models.stylegan2.op import fused_leaky_relu, upfirdn2d


class ScaledLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True, scale: float=1.0):
        super().__init__()

        self.scale = scale
        self.weight = nn.Parameter(torch.rand(out_features, in_features) * 2 - 1)

        if bias:
            self.bias = nn.Parameter(nn.Linear(in_features, out_features).bias.data)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, weight=self.weight * self.scale, bias=self.bias)


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization by @Kaixhin
    https://github.com/pytorch/pytorch/issues/8985
    """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)

        return out


class Reshape(nn.Module):
    def __init__(self, target_shape: Tuple[int]):
        super(Reshape, self).__init__()

        self.target_shape = target_shape

    def forward(self, x):
        return x.view(*self.target_shape)


class ConvMeanPool(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding=0, use_sn: bool=False):
        super(ConvMeanPool, self).__init__(
            sn_wrapper(nn.Conv2d(in_channels, out_channels, kernel_size, bias, padding=padding), use_sn),
            nn.AvgPool2d((2,2)),
        )


class MeanPoolConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding=0, use_sn: bool=False):
        super(MeanPoolConv, self).__init__(
            nn.AvgPool2d((2,2)),
            sn_wrapper(nn.Conv2d(in_channels, out_channels, kernel_size, bias, padding=padding), use_sn),
        )


class DepthToSpace(nn.Module):
    """
    Copy-pasted from https://github.com/takuhirok/NR-GAN/blob/master/models/common.py
    """
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_square = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, in_height, in_width, in_depth) = output.size()
        out_depth = int(in_depth / self.block_size_square)
        out_width = int(in_width * self.block_size)
        out_height = int(in_height * self.block_size)
        output = output.contiguous().view(
            batch_size, in_height, in_width, self.block_size_square, out_depth)
        output_list = output.split(self.block_size, 3)
        output_list = [
            output_element.contiguous().view(batch_size, in_height, out_width, out_depth)
            for output_element in output_list
        ]
        output = torch.stack(output_list, 0) \
                        .transpose(0, 1) \
                        .permute(0, 2, 1, 3, 4).contiguous() \
                        .view(batch_size, out_height, out_width, out_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):
    """
    Copy-pasted from https://github.com/takuhirok/NR-GAN/blob/master/models/common.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding=0, use_sn: bool=False):
        super(UpSampleConv, self).__init__()
        self.conv = sn_wrapper(nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, padding=padding), use_sn)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)

        return output


class OptimizedResidualBlock(nn.Module):
    """
    Copy-pasted from https://github.com/takuhirok/NR-GAN/blob/master/models/common.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, main_branch_weight=0.0, use_sn: bool=False):
        super(OptimizedResidualBlock, self).__init__()

        self.main_branch_weight = nn.Parameter(torch.tensor(main_branch_weight))
        self.conv1 = sn_wrapper(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1), use_sn)
        self.conv2 = ConvMeanPool(out_channels, out_channels, kernel_size, padding=1, use_sn=use_sn)
        self.conv_shortcut = MeanPoolConv(in_channels, out_channels, 1, padding=0, use_sn=use_sn)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        shortcut = self.conv_shortcut(x)

        output = x
        output = self.conv1(output)
        output = self.relu2(output)
        output = self.conv2(output)

        return shortcut + self.main_branch_weight * output


class ResidualBlock(nn.Module):
    """
    Copy-pasted from https://github.com/takuhirok/NR-GAN/blob/master/models/common.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, resample=None, main_branch_weight=0.0, use_sn: bool=False):
        super(ResidualBlock, self).__init__()
        self.main_branch_weight = nn.Parameter(torch.tensor(main_branch_weight))

        if in_channels != out_channels or resample is not None:
            self.learnable_shortcut = True
        else:
            self.learnable_shortcut = False

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        if resample == 'down':
            self.conv_shortcut = ConvMeanPool(in_channels, out_channels, 1, padding=0, use_sn=use_sn)
            self.conv1 = sn_wrapper(nn.Conv2d(in_channels, in_channels, kernel_size, padding=1), use_sn)
            self.conv2 = ConvMeanPool(in_channels, out_channels, kernel_size, padding=1, use_sn=use_sn)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(in_channels, out_channels, 1, padding=0, use_sn=use_sn)
            self.conv1 = UpSampleConv(in_channels, out_channels, kernel_size, padding=1, use_sn=use_sn)
            self.conv2 = sn_wrapper(nn.Conv2d(out_channels, out_channels, kernel_size, padding=1), use_sn)
        elif resample is None:
            if self.learnable_shortcut:
                self.conv_shortcut = sn_wrapper(nn.Conv2d(in_channels, out_channels, 1, padding=0), use_sn)

            self.conv1 = sn_wrapper(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1), use_sn)
            self.conv2 = sn_wrapper(nn.Conv2d(out_channels, out_channels, kernel_size, padding=1), use_sn)
        else:
            raise NotImplementedError(f'Unknown resample type: {resample}')

    def forward(self, input):
        if self.learnable_shortcut:
            shortcut = self.conv_shortcut(input)
        else:
            shortcut = input

        output = input
        output = self.relu1(output)
        output = self.conv1(output)
        output = self.relu2(output)
        output = self.conv2(output)

        return shortcut + self.main_branch_weight * output


def create_activation(activation_type: str, *args, **kwargs) -> nn.Module:
    if activation_type == 'leaky_relu':
        return nn.LeakyReLU(*args, **kwargs)
    elif activation_type == 'scaled_leaky_relu':
        return ScaledLeakyReLU(*args, **kwargs)
    elif activation_type == 'relu':
        return nn.ReLU(*args, **kwargs)
    elif activation_type == 'tanh':
        return nn.Tanh(*args, **kwargs)
    elif activation_type == 'none' or activation_type is None:
        return nn.Identity(*args, **kwargs)
    elif activation_type == 'centered_sigmoid':
        return CenteredSigmoid(*args, **kwargs)
    elif activation_type == 'sine':
        return Sine(*args, **kwargs)
    elif activation_type == 'unit_centered_sigmoid':
        return UnitCenteredSigmoid(*args, **kwargs)
    elif activation_type == 'unit_centered_tanh':
        return UnitCenteredTanh(*args, **kwargs)
    elif activation_type == 'sines_cosines':
        return SinesCosines(*args, **kwargs)
    elif activation_type == 'normalizer':
        return Normalizer(*args, **kwargs)
    elif activation_type == 'normal_dist_uniformer':
        return NormalDistUniformer(*args, **kwargs)
    elif activation_type == 'normal_clip':
        return NormalClip(*args, **kwargs)
    else:
        raise NotImplementedError(f'Unknown activation type: {activation_type}')


class CenteredSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid() * 2 - 1


class UnitCenteredSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid() * 2


class UnitCenteredTanh(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh() + 1


class Sine(nn.Module):
    def __init__(self, scale: float=1.0):
        super(Sine, self).__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        return x.sin()


class Normalizer(nn.Module):
    """
    Just normalizes its input
    """
    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2, f"Wrong shape: {x.shape}"

        return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)


class NormalClip(nn.Module):
    """
    Clips input values into [-2, 2] region
    """
    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, -2, 2)


class NormalDistUniformer(nn.Module):
    """
    Transforms normal distribution into uniform in [-scale, scale] range
    Using https://en.wikipedia.org/wiki/Normal_distribution#Numerical_approximations_for_the_normal_CDF
    """
    def __init__(self, initial_output_scale: float=np.sqrt(3)):
        super(NormalDistUniformer, self).__init__()

        b_coefs = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]

        self.register_buffer('b_coefs', torch.tensor(b_coefs).view(1, 1, 5))
        self.register_buffer('powers', torch.arange(1, len(b_coefs) + 1).view(1, 1, 5))
        self.output_scale = nn.Parameter(torch.tensor(initial_output_scale))

    def compute_cdf(self, x: Tensor) -> Tensor:
        assert x.ndim == 2, f"Wrong shape: {x.shape}"

        x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True) # [num_inrs, batch_size]
        pdfs = (-0.5 * x.pow(2)).exp() / np.sqrt(2 * np.pi) # [num_inrs, batch_size]
        t = 1 / (1 + 0.2316419 * x.abs()) # [num_inrs, batch_size]
        t_pow = t.unsqueeze(2).repeat(1, 1, 5) ** self.powers # [num_inrs, batch_size, 5]
        cdfs = 1 - pdfs * (t_pow * self.b_coefs).sum(dim=2) # [num_inrs, batch_size]
        cdfs[x < 0] = -cdfs[x < 0] + 1

        return cdfs

    def forward(self, x: Tensor) -> Tensor:
        cdfs = self.compute_cdf(x) # they now lie in [0, 1] range
        result = (cdfs * 2 - 1) * self.output_scale # [-scale, scale] range

        return result


class LinearResidual(nn.Module):
    def __init__(self, config: Config, transform: Callable):
        super().__init__()

        self.config = config
        self.transform = transform
        self.weight = nn.Parameter(
            torch.tensor(config.weight).float(), requires_grad=config.learnable_weight)

    def forward(self, x: Tensor) -> Tensor:
        if self.config.weighting_type == 'shortcut':
            return self.transform(x) + self.weight * x
        elif self.config.weighting_type == 'residual':
            return self.weight * self.transform(x) + x
        else:
            raise ValueError


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        # self.register_buffer('scale', torch.tensor(1 / math.sqrt(in_channel * kernel_size ** 2)))

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, additional_scale:float=1.0):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul * additional_scale
        # self.register_buffer('scale', torch.tensor((1 / math.sqrt(in_dim)) * lr_mul))
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float=0.2, scale: float=None):
        super().__init__()

        self.negative_slope = negative_slope
        self.scale = math.sqrt(2 / (1 + negative_slope ** 2)) if scale is None else scale

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * self.scale


class SinesCosines(nn.Module):
    """
    Sines-cosines activation function
    It applies both sines and cosines and concatenates the results
    """
    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x.sin(), x.cos()], dim=1)


class SizeSampler(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.stats_path = self.config.hp.generator.size_sampler.stats_path
        self.stats = np.load(self.stats_path, allow_pickle=True)
        self.pos_embedder = nn.Sequential(
            nn.Linear(1, self.config.hp.generator.size_sampler.pos_emb_dim // 2),
            SinesCosines()
        )

    def sample_aspect_ratios(self, labels: Tensor) -> Tensor:
        """
        Given class idx, samples two things:
            - list of sizes
            - aspect ratio features
        """
        sizes = [random.sample(self.stats[c], 1)[0] for c in labels]
        aspect_ratios = torch.tensor([(s[0] / s[1]) for s in sizes]).to(labels.device)

        return aspect_ratios

    def sample_labels(self, batch_size: int) -> Tensor:
        return torch.randint(low=0, high=self.config.data.num_classes, size=(batch_size,))


class BlurUpsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


def blur_upsample(x: Tensor, kernel: Tensor, upsample_factor: int):
    kernel = make_kernel(kernel).to(x.device) * (upsample_factor ** 2)
    p = kernel.shape[0] - upsample_factor
    pad0 = (p + 1) // 2 + upsample_factor - 1
    pad1 = p // 2
    pad = (pad0, pad1)

    return upfirdn2d(x, kernel, up=upsample_factor, down=1, pad=pad)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


def sn_wrapper(module: nn.Module, use_sn: bool, *sn_args, **sn_kwargs) -> nn.Module:
    """
    So not to wrap it everywhere
    """
    if use_sn:
        return nn.utils.spectral_norm(module, *sn_args, **sn_kwargs)
    else:
        return module
