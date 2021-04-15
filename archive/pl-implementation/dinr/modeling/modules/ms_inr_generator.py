import math
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .conv import StyledConv
from .linear import EqualLinear, StyledLinear
from .style import PixelNorm, CoordBasedConstantInput, CoordFuser
from .torgb import ToRGB


class MSINRGenerator(nn.Module):
    def __init__(self,
        fallback: bool,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=(1, 3, 3, 1),
        lr_mlp=0.01,
        transform_type='linear',
        modulation_type='fmm_inrgan',
        is_multiscale: bool=False,
        interpolation_type='nearest',
        factorization_rank: int=3,
        fourier_scale: int=10.0,
        coord_fuse_type: str="concat",
        multi_scale_coord_embs: bool=False,
        use_noise_injection: bool=True,
        coord_emb_dim_multiplier: int=0.5,
        linear_conv_block: bool=False,
        no_fourier_embs: bool=False):

        super().__init__()

        self.size = size
        self.style_dim = style_dim
        self.is_multiscale = is_multiscale
        self.interpolation_type = interpolation_type

        mapping_network_layers = [PixelNorm()]

        for i in range(n_mlp):
            mapping_network_layers.append(EqualLinear(
                self.style_dim, self.style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))

        self.style = nn.Sequential(*mapping_network_layers)
        self.log_size = int(math.log(self.size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        if not fallback:
            if self.is_multiscale:
                self.channels = {
                    4: 512,
                    8: 512,
                    16: 512,
                    32: 512,
                    64: int(256 * channel_multiplier),
                    128: int(128 * channel_multiplier),
                    256: int(64 * channel_multiplier),
                    512: int(32 * channel_multiplier),
                    1024: int(16 * channel_multiplier),
                }
                self.coord_concat_emb_dims = {
                    4: int(128 * coord_emb_dim_multiplier),
                    8: int(128 * coord_emb_dim_multiplier),
                    16: int(128 * coord_emb_dim_multiplier),
                    32: int(256 * coord_emb_dim_multiplier),
                    64: int(256 * channel_multiplier * coord_emb_dim_multiplier),
                    128: int(128 * channel_multiplier * coord_emb_dim_multiplier),
                    256: int(64 * channel_multiplier * coord_emb_dim_multiplier),
                    512: int(32 * channel_multiplier * coord_emb_dim_multiplier),
                    1024: int(16 * channel_multiplier * coord_emb_dim_multiplier),
                }
                resolutions = [2 ** log_res for log_res in range(3, self.log_size + 1)]
                input_resolution = 4
                use_coordinates_input = False
            else:
                self.channels = {256: 512}
                resolutions = [256] * len(range(3, self.log_size + 1))
                input_resolution = 256
                use_coordinates_input = True

            kernel_size = 1
            use_upsample = False

            factorization_ranks = {
                4: 3,
                8: 3,
                16: 3,
                32: 5,
                64: 5,
                128: 5,
                256: 10,
                512: 10,
                1024: 10,
            }
        else:
            self.channels = {
                4: 512,
                8: 512,
                16: 512,
                32: 512,
                64: 256 * channel_multiplier,
                128: 128 * channel_multiplier,
                256: 64 * channel_multiplier,
                512: 32 * channel_multiplier,
                1024: 16 * channel_multiplier,
            }
            resolutions = [2 ** log_res for log_res in range(3, self.log_size + 1)]
            kernel_size = 3
            input_resolution = 4
            use_upsample = True
            use_coordinates_input = False
            use_noise_injection = True

        self.input = CoordBasedConstantInput(self.channels[input_resolution], input_resolution, use_coordinates_input)

        if fallback or transform_type == 'conv':
            self.conv1 = StyledConv(
                self.input.get_total_dim(),
                self.channels[input_resolution],
                kernel_size,
                self.style_dim,
                blur_kernel=blur_kernel,
                noise_injection=use_noise_injection)
        else:
            self.conv1 = StyledLinear(
                self.input.get_total_dim(),
                self.channels[input_resolution],
                self.style_dim,
                modulation_type,
                factorization_ranks[input_resolution])

        self.to_rgb1 = ToRGB(self.channels[input_resolution], self.style_dim, upsample=False)
        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        self.coord_concats = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        in_channel = self.channels[resolutions[0]]

        for res in resolutions:
            out_channel = self.channels[res]

            if self.is_multiscale:
                self.coord_concats.append(CoordFuser(
                    emb_dim=self.coord_concat_emb_dims[res],
                    style_dim=self.style_dim,
                    coord_dim=2,
                    fourier_scale=fourier_scale,
                    fuse_type=coord_fuse_type,
                    multi_scale_coord_embs=multi_scale_coord_embs,
                    img_size=res,
                    no_fourier_embs=no_fourier_embs,
                ))

                in_channel += self.coord_concats[-1].get_total_dim()
            else:
                self.coord_concats.append(nn.Identity())

            if fallback or transform_type == 'conv':
                self.convs.append(StyledConv(
                    in_channel,
                    out_channel,
                    kernel_size,
                    self.style_dim,
                    upsample=use_upsample,
                    blur_kernel=blur_kernel,
                    noise_injection=use_noise_injection
                ))
                self.convs.append(StyledConv(
                    out_channel,
                    out_channel,
                    kernel_size,
                    self.style_dim,
                    blur_kernel=blur_kernel,
                    noise_injection=use_noise_injection))
            elif transform_type == 'linear':
                self.convs.append(StyledLinear(in_channel, out_channel, self.style_dim, modulation_type, factorization_ranks[res]))

                if linear_conv_block:
                    self.convs.append(StyledConv(
                        out_channel,
                        out_channel,
                        3,
                        self.style_dim,
                        blur_kernel=blur_kernel,
                        noise_injection=use_noise_injection))
                else:
                    self.convs.append(StyledLinear(out_channel, out_channel, self.style_dim, modulation_type, factorization_ranks[res]))
            else:
                raise NotImplementedError(f'Unknown transform type: {transform_type}')

            self.to_rgbs.append(ToRGB(out_channel, self.style_dim, upsample=use_upsample))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def forward(
            self,
            styles,
            return_latents=False,
            inject_index=None,
            truncation: float=1.0,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
    ):
        if isinstance(styles, torch.Tensor):
            styles = [styles]

        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]

        if truncation < 1.0:
            style_t = []
            for style in styles:
                style_t.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb, coord_concat in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs, self.coord_concats):

            if self.is_multiscale:
                out = F.interpolate(out, scale_factor=2, mode=self.interpolation_type) # [batch_size, c, h * 2, w * 2]
                skip = F.interpolate(skip, scale_factor=2, mode=self.interpolation_type) # [batch_size, 3, h * 2, w * 2]
                out = coord_concat(out, latent[:, i]) # [batch_size, c + emb_dim, h, w]
            out = conv1(out, latent[:, i], noise=noise1) # [batch_size, c, h, w]
            out = conv2(out, latent[:, i + 1], noise=noise2) # [batch_size, c, h, w]
            skip = to_rgb(out, latent[:, i + 2], skip) # [batch_size, 3, h * 2, w * 2]

            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None
