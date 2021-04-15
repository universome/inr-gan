"""
Adapted from https://github.com/rosinality/stylegan2-pytorch/blob/master/calc_inception.py
"""
import os
import sys
from pathlib import Path

import torch
from torch import nn
import hydra
from omegaconf import DictConfig
import numpy as np
from tqdm import tqdm

from dinr.modeling.metrics.inception import InceptionV3
from dinr.data.build import build_datasets, build_loaders

sys.path.extend(['..'])
from configs import trainer_conf


@torch.no_grad()
def extract_features(dataloader, inception, device):
    feature_list = []

    for batch in tqdm(dataloader):
        img = batch['img'].to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to("cpu"))

    features = torch.cat(feature_list, 0)

    return features


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    datasets = build_datasets(cfg, 'test')
    dataloaders = build_loaders(cfg, datasets, 'test')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception = InceptionV3([3], normalize_input=False)
    inception = nn.DataParallel(inception).eval().to(device)

    for i, (dataset_name, dataset_param, _) in enumerate(datasets):
        print(f"Processing {dataset_name} dataset...")

        path = dataset_param.inception_real_stats_path
        if os.path.exists(path):
            print(f"Stats for {dataset_name} already exist at {path}. Skipping.")
            continue
        print(f"Stats will be saved at {path}")
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

        features = extract_features(dataloaders[i], inception, device).numpy()
        print(f"Extracted {features.shape[0]} features")

        mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False)
        np.savez(path, mean=mean, cov=cov)

        print(f"Stats for {dataset_name} are saved at {path}")


if __name__ == "__main__":
    main()
