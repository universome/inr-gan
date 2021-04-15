import random
from typing import Optional

import torch
from torch import nn

from .conv import StyledConv
from .linear import EqualLinear
from .positional_encoding import FourierMapping, CoordinateEmbedding
from .style import PixelNorm
from .torgb import ToRGB


class INRGenerator(nn.Module):
    def __init__(
            self,
            style_dim,
            n_mlp,
            num_channels=512,
            num_layers=15,
            fourier_mapping: Optional[FourierMapping] = None,
            coordinate_embedding: Optional[CoordinateEmbedding] = None,
            lr_mlp: float = 0.01,
            factorization_rank: Optional[int] = None,
            use_sigmoid: bool = False,
            skip_connections: bool = False
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_layers = num_layers
        self.style_dim = style_dim
        self.skip_connections = skip_connections

        assert fourier_mapping is not None or coordinate_embedding is not None
        self.fourier_mapping = fourier_mapping
        self.coordinate_embedding = coordinate_embedding

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"))
        self.style = nn.Sequential(*layers)

        in_features = 0
        if fourier_mapping is not None:
            in_features += fourier_mapping.out_features
        if coordinate_embedding is not None:
            in_features += coordinate_embedding.embedding_dim

        self.conv1 = StyledConv(in_features, self.num_channels, 1, style_dim,
                                factorization_rank=factorization_rank, use_sigmoid=use_sigmoid,
                                noise_injection=False)

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        for i in range(self.num_layers // 2):
            self.convs.append(StyledConv(self.num_channels, self.num_channels, 1, style_dim,
                                         factorization_rank=factorization_rank, use_sigmoid=use_sigmoid,
                                         noise_injection=False))
            self.convs.append(StyledConv(self.num_channels, self.num_channels, 1, style_dim,
                                         factorization_rank=factorization_rank, use_sigmoid=use_sigmoid,
                                         noise_injection=False))
            if skip_connections or i == self.num_layers // 2 - 1:
                self.to_rgbs.append(ToRGB(self.num_channels, style_dim, upsample=False))

        self.n_latent = self.num_layers + 1

    # def mean_latent(self, n_latent):
    #     latent_in = torch.randn(
    #         n_latent, self.style_dim, device=self.input.input.device
    #     )
    #     latent = self.style(latent_in).mean(0, keepdim=True)
    #
    #     return latent
    #
    # def get_latent(self, input):
    #     return self.style(input)

    def forward(
            self,
            grid,
            styles,
            return_latents=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            return_embeddings=False
    ):
        """
        Args:
            grid (Tensor): of shape (B, N, C) e.g. C = 2 if (x,y)
        Returns:
            Tensor: of shape (B, 3, 1, N) and latents if return_latents=True
        """
        if isinstance(styles, torch.Tensor):
            styles = [styles]
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if truncation < 1:
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

        batch_size, num_points, in_dim = grid.size()
        input = []

        if self.fourier_mapping is not None:
            fourier_features = self.fourier_mapping(grid.reshape(batch_size * num_points, in_dim))
            fourier_features = fourier_features.view(batch_size, num_points, -1)
            fourier_features = fourier_features.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, num_points)
            input.append(fourier_features)
        if self.coordinate_embedding is not None:
            if in_dim == 2:
                co_features = self.coordinate_embedding(grid.view(batch_size, 1, num_points, 2), return_embeddings)
            elif in_dim == 3:
                co_features = self.coordinate_embedding(grid.view(batch_size, 1, 1, num_points, 3), return_embeddings)
                co_features = co_features.squeeze(2)
            else:
                raise NotImplementedError
            # co_features: Tensor of shape (B, C, 1, num_points)
            input.append(co_features)

        input = torch.cat(input, dim=1)

        out = self.conv1(input, latent[:, 0])
        skip = torch.zeros((out.size(0), 3, 1, out.size(3)), device=out.device)
        i = 1

        if self.skip_connections:
            for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2], self.to_rgbs):
                out = conv1(out, latent[:, i])
                out = conv2(out, latent[:, i + 1])
                skip = to_rgb(out, latent[:, i + 2], skip)
                i += 2
            result = skip
        else:
            for conv in self.convs:
                out = conv(out, latent[:, i])
                i += 1
            result = self.to_rgbs[-1](out, latent[:, i], skip)

        if return_latents:
            return result, latent
        else:
            return result, None
