"""
This script computes imgs/sec for a generator in the eval mode
for different batch sizes
"""
import sys; sys.path.extend(['..', '.', 'src'])
import time

import numpy as np
import torch
import torch.nn as nn
import hydra
from hydra.experimental import compose, initialize
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch.autograd.profiler as profiler

from src import dnnlib


def instantiate_G(cfg: DictConfig) -> nn.Module:
    hydra_cfg = compose(config_name=f'../../configs/{cfg.model}.yml')

    if cfg.model in ['inr-gan', 'inr-gan-bil', 'cips']:
        hydra_cfg.generator.coords.use_full_cache = True

    G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    G_kwargs.synthesis_kwargs.channel_base = int(hydra_cfg.generator.get('fmaps', 0.5) * 32768)
    G_kwargs.synthesis_kwargs.channel_max = 512
    G_kwargs.mapping_kwargs.num_layers = hydra_cfg.generator.get('mapping_net_n_layers', 2)
    if cfg.get('num_fp16_res', 0) > 0:
        G_kwargs.synthesis_kwargs.num_fp16_res = cfg.num_fp16_res
        G_kwargs.synthesis_kwargs.conv_clamp = 256
    G_kwargs.cfg = OmegaConf.to_container(hydra_cfg.generator)
    G_kwargs.c_dim = 0
    G_kwargs.img_resolution = cfg.get('resolution', 256)
    G_kwargs.img_channels = 3

    G = dnnlib.util.construct_class_by_name(**G_kwargs).eval().requires_grad_(False).to(cfg.device)

    return G


@torch.no_grad()
def profile_for_batch_size(G: nn.Module, cfg: DictConfig, batch_size: int):
    z = torch.randn(batch_size, G.z_dim, device=cfg.device)
    c = None
    times = []

    for i in tqdm(range(cfg.num_warmup_iters), desc='Warming up'):
        torch.cuda.synchronize()
        fake_img = G(z, c).contiguous()
        y = fake_img[0, 0, 0, 0].item() # sync
        torch.cuda.synchronize()

    time.sleep(1)

    torch.cuda.reset_peak_memory_stats()

    with profiler.profile(record_shapes=True, use_cuda=True) as prof:
        for i in tqdm(range(cfg.num_profile_iters), desc='Profiling'):
            torch.cuda.synchronize()
            start_time = time.time()
            with profiler.record_function("forward"):
                fake_img = G(z, c).contiguous()
                y = fake_img[0, 0, 0, 0].item() # sync
            torch.cuda.synchronize()
            times.append(time.time() - start_time)

    torch.cuda.empty_cache()
    num_imgs_processed = len(times) * batch_size
    total_time_spent = np.sum(times)
    bandwidth = num_imgs_processed / total_time_spent
    summary = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)

    print(f'[Batch size: {batch_size}] Mean: {np.mean(times):.05f}s/it. Std: {np.std(times):.05f}s')
    print(f'[Batch size: {batch_size}] Imgs/sec: {bandwidth:.03f}')
    print(f'[Batch size: {batch_size}] Max mem: {torch.cuda.max_memory_allocated(cfg.device) / 2**30:<6.2f} gb')

    return bandwidth, summary


@hydra.main(config_name="../../configs/profile.yml")
def profile(cfg: DictConfig):
    G = instantiate_G(cfg)
    bandwidths = []
    summaries = []
    print(f'Number of parameters: {sum(p.numel() for p in G.parameters())}')

    for batch_size in cfg.batch_sizes:
        bandwidth, summary = profile_for_batch_size(G, cfg, batch_size)
        bandwidths.append(bandwidth)
        summaries.append(summary)

    best_batch_size_idx = int(np.argmax(bandwidths))
    print(f'------------ Best batch size for {cfg.model} is {cfg.batch_sizes[best_batch_size_idx]} ------------')
    print(summaries[best_batch_size_idx])


if __name__ == '__main__':
    profile()
