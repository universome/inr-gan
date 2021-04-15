import logging
import os

import numpy as np
import torch
from pytorch_lightning.metrics import Metric
from scipy import linalg

from dinr.utils.comm import is_main_process

log = logging.getLogger(__name__)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        log.info(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


class FID(Metric):
    def __init__(self, real_stats_path: os.PathLike, dims: int = 2048):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.dims = dims
        self.add_state("fake_activations", default=torch.zeros(0, dims), dist_reduce_fx=None)
        self.add_state("real_activations", default=torch.zeros(0, dims), dist_reduce_fx=None)
        self.real_stats_path = real_stats_path

        if os.path.exists(real_stats_path):
            self.load_real_stats()
        else:
            print(f"Real stats are not found at {real_stats_path}. FID will not be calculated.")
            # TODO(universome): put the stats for popular datasets
            # on some cloud storage and download them here
            self.real_stats = None

    def update(self, inception_features: torch.Tensor, images_are_real: bool):
        if images_are_real:
            self.real_activations = torch.cat(
                [self.real_activations.cpu(), inception_features.detach().cpu()], dim=0)
        else:
            self.fake_activations = torch.cat(
                [self.fake_activations.cpu(), inception_features.detach().cpu()], dim=0)

    def save_real_stats(self):
        if is_main_process():
            os.makedirs(os.path.dirname(self.real_stats_path), exist_ok=True)
            np.savez(self.real_stats_path[:-4], mean=self.real_stats[0], cov=self.real_stats[1])

    def load_real_stats(self):
        real_stats = np.load(self.real_stats_path)
        self.real_stats = (real_stats['mean'], real_stats['cov'])

    def compute(self):
        if self.real_stats is None:
            return np.nan

        self.real_activations = self.real_activations.cpu()
        self.fake_activations = self.fake_activations.cpu()

        if self.real_activations.ndim == 3:
            num_devices, b, feat_dim = self.real_activations.shape
            self.real_activations = self.real_activations.view(num_devices * b, feat_dim)
        if self.fake_activations.ndim == 3:
            num_devices, b, feat_dim = self.fake_activations.shape
            self.fake_activations = self.fake_activations.view(num_devices * b, feat_dim)

        if self.real_stats is None:
            m_real, s_real = calculate_activation_statistics(self.real_activations.numpy())
            self.real_stats = (m_real, s_real)
            self.save_real_stats()
            self.real_activations = self.real_activations.new_zeros(0, self.dims)
        else:
            m_real, s_real = self.real_stats
        m_fake, s_fake = calculate_activation_statistics(self.fake_activations.numpy())
        fid_value = calculate_frechet_distance(m_real, s_real, m_fake, s_fake)
        return fid_value
