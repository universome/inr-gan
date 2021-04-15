"""Generates a dataset of images using pretrained network pickle."""

import sys; sys.path.extend(['.', 'src'])
import os
import re
import random
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

import legacy

torch.set_grad_enabled(False)


#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network_pkl', help='Network pickle filename', required=True)
@click.option('--truncation_psi', type=float, help='Truncation psi', default=1.0, show_default=True)
@click.option('--external_truncation_psi', type=float, help='External truncation psi (no w_avg)', default=1.0, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--num_images', type=int, help='Number of images to generate', default=50000, show_default=True)
@click.option('--batch_size', type=int, help='Batch size to use for generation', default=32, show_default=True)
@click.option('--seed', type=int, help='Random seed', default=42, metavar='DIR')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    truncation_psi: float,
    external_truncation_psi: float,
    noise_mode: str,
    num_images: int,
    batch_size: int,
    seed: int,
    outdir: str,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore

    os.makedirs(outdir, exist_ok=True)

    if external_truncation_psi < 1:
        z = torch.randn(10000, G.z_dim, device=device) # [10000, z_dim]
        w = G.mapping(z, None) # [10000, num_ws, w_dim]
        w_avg_ext = w[:, 0].mean(dim=0, keepdim=True).unsqueeze(1) # [1, 1, w_dim]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate images.
    for batch_idx in tqdm(range((num_images + batch_size - 1) // batch_size)):
        z = torch.randn(batch_size, G.z_dim, device=device) # [batch_size, z_dim]
        w = G.mapping(z, None, truncation_psi=truncation_psi) # [batch_size, num_ws, z_dim]
        if external_truncation_psi < 1:
            w = (1 - external_truncation_psi) * w_avg_ext + external_truncation_psi * w
        imgs = G.synthesis(w, noise_mode=noise_mode)

        # z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        # img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        for i, img in enumerate(imgs):
            image_num = batch_idx * batch_size + i
            if image_num >= num_images:
                break
            PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(f'{outdir}/img_{image_num:06d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
