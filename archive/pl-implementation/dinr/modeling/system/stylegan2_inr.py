from typing import Optional

import torch

from .stylegan2 import StyleGAN2


class StyleGAN2INR(StyleGAN2):
    def forward(self, z, grid: Optional[torch.Tensor] = None, ndim=2, return_latents=False, **generator_forward_kwargs):
        if grid is None:
            if ndim == 2:
                size = 256  # todo pass it using config
                x, y = torch.linspace(-1, 1, size, device=self.device), torch.linspace(-1, 1, size, device=self.device)
                grid_y, grid_x = torch.meshgrid(y, x)
                grid_xy = torch.stack([grid_x, grid_y], dim=2).view(1, -1, 2)
                batch_size = z.shape[0] if isinstance(z, torch.Tensor) else z[0].shape[0]
                grid_xy = grid_xy.expand(batch_size, grid_xy.size(1), 2)
                grid = grid_xy
                # grid = grid_xy.contiguous()
            else:
                raise NotImplementedError

        if self.training:
            fake_img, latent = self.generator(grid, z, return_latents=return_latents, return_embeddings=True,
                                              **generator_forward_kwargs)
        else:
            fake_img, latent = self.generator_ema(grid, z, return_latents=return_latents, **generator_forward_kwargs)
            fake_img = fake_img.clamp(-1.0, 1.0)

        if ndim == 2:
            # fake_img of shape (B, 3, 1, N)
            batch_size = fake_img.size(0)
            fake_img = fake_img.view(batch_size, 3, 256, 256)  # todo pass value 256 using config
        else:
            raise NotImplementedError

        if return_latents:
            return fake_img, latent

        return fake_img
