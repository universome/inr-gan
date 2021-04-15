import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch_lightning.metrics import Metric

log = logging.getLogger(__name__)


from .lpips_models import VGG16, LinLayers


class LPIPS(Metric):
    """
    Nearest neighbour LPIPS for generative models to check for overfitting

    Adapted from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
    Uses VGG16 under the hood with those parameters that are provided by the authors:
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/vgg.pth
    """
    def __init__(self, num_fake_images: int):
        """
        Args: num_fake_images — number of fake images to find a nearest neighbour to.
        """
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        self.feat_extractor = VGG16()
        self.dist_scaler = LinLayers(self.feat_extractor.n_channels_list)
        self.dist_scaler.load_weights("vgg16")
        self.add_state("min_dists", default=torch.ones(num_fake_images, 1) * 1000.0, dist_reduce_fx=None)

    def extract_features(self, images: Tensor) -> List[Tensor]:
        self.model.eval()
        self.dist_scaler.eval()

        with torch.no_grad():
            feats = self.feat_extractor(images)

        return feats

    def update(self, fake_images: Tensor, real_images: Tensor):
        new_min_dists = self.compute_min_dists(fake_images, real_images).cpu()
        self.min_dists = torch.stack([self.min_dists.cpu(), new_min_dists], dim=0).min(dim=0)

    @torch.no_grad()
    def compute_min_dists(self, images_fake: Tensor, images_real: Tensor) -> Tensor:
        """
        Returns [num_fake_images x num_real_images] distance matrix
        Assumes that images are in [-1, 1] range
        """
        feats_fake = self.extract_features(images_fake) # [num_layers, num_fake_images, c_l, h_l, w_l]
        feats_real = self.extract_features(images_real) # [num_layers, num_real_images, c_l, h_l, w_l]

        distances = []

        for i in len(feats_fake[0]):
            curr_fake_feats = [f[i].unsqueeze(0) for f in feats_fake] # [num_layers, 1, c_l, h_l, w_l]
            diff = [(ff - fr) ** 2 for ff, fr in zip(curr_fake_feats, feats_real)] # [num_layers, num_real_images, c_l, h_l, w_l]
            scaled_diff = [l(d).mean(dim=(2, 3)).squeeze(2) for d, l in zip(diff, self.dist_scaler)] # [num_layers, num_real_images]
            lpips = torch.cat(scaled_diff, dim=0).sum(dim=0) # [num_real_images]
            distances.append(num_real_images)

        distances = torch.stack(distances) # [num_fake_images, num_real_images]
        min_dists = distances.min(dim=1) # [num_fake_images]

        return min_dists

    def compute(self):
        return self.min_dists.mean()
