#!/usr/bin/env python3
"""
Copy-pasted from: https://raw.githubusercontent.com/mseitzer/pytorch-fid/master/fid_score.py

Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from typing import List
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from scipy import linalg
from PIL import Image

from .inception import InceptionV3


def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


class FilesDataset:
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx) -> np.array:
        img = imread(file_paths[id])
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        img /= 255

        return img


def compute_feats(dataset, model, batch_size=64, device: str='cpu', is_ds_labeled: bool=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- dataset     : Dataset of images
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- device      : Which device to run inference on

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(dataset):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(dataset)

    result = []
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=4)

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0] if is_ds_labeled else batch # Ignoring labels
            images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
            images = images.to(device)

            feats = model(images)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if feats.size(2) != 1 or feats.size(3) != 1:
                feats = adaptive_avg_pool2d(feats, output_size=(1, 1))

            result.extend(feats.cpu().data.numpy().reshape(feats.size(0), -1))

    return np.array(result)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6, decomposed: bool=False):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is:

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

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    if decomposed:
        return {
            'total': fid,
            'means_diff': diff.dot(diff),
            'trace_a': np.trace(sigma1),
            'trace_b': np.trace(sigma2),
            'cov_mean': -2 * tr_covmean
        }
    else:
        return fid


def calculate_frechet_distance_torch(mu1, sigma1, mu2, sigma2, eps=1e-6, decomposed: bool=False, num_approx_iters: int=50):
    """
    Taken from https://github.com/ajbrock/BigGAN-PyTorch
    (who took it from https://github.com/bioinf-jku/TTUR)
    """
    # Convert just in case numpy arrays are given as input
    if isinstance(mu1, np.ndarray): mu1 = torch.from_numpy(mu1)
    if isinstance(sigma1, np.ndarray): sigma1 = torch.from_numpy(sigma1)
    if isinstance(mu2, np.ndarray): mu2 = torch.from_numpy(mu2)
    if isinstance(sigma2, np.ndarray): sigma2 = torch.from_numpy(sigma2)

    assert mu1.shape == mu2.shape, f'Wrong shapes: {mu1.shape} vs {mu2.shape}'
    assert sigma1.shape == sigma2.shape, f'Wrong shapes: {sigma1.shape} vs {sigma2.shape}'

    diff = mu1 - mu2
    covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), num_approx_iters).squeeze()
    fid = (diff.dot(diff) +  torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean))

    if decomposed:
        return {
            'total': fid,
            'means_diff': diff.dot(diff),
            'trace_a': torch.trace(sigma1),
            'trace_b': torch.trace(sigma2),
            'cov_mean': -2 * torch.trace(covmean)
        }
    else:
        return fid


# Copy-pasted from https://github.com/ajbrock/BigGAN-PyTorch/blob/master/inception_utils.py
# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt
def sqrt_newton_schulz(A, num_approx_iters, dtype=None):
    if dtype is None:
        dtype = A.type()
    batchSize = A.shape[0]
    dim = A.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
    I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    Z = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    for i in range(num_approx_iters):
        T = 0.5*(3.0*I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)

    return sA


def compute_statistics_for_dataset(dataset, model, batch_size=50, device: str='cpu', is_ds_labeled: bool=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- dataset      : Dataset of images (as transposed numpy arrays)
    -- model        : Instance of inception model
    -- batch_size   : The images numpy array is split into batches with
                      batch size batch_size. A reasonable batch size
                      depends on the hardware.
    -- is_ds_labeled: flag denoting if the dataset is labeled
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    feats = compute_feats(dataset, model, batch_size, device, is_ds_labeled)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)

    return mu, sigma


def compute_statistics_for_path(path, model, batch_size, device):
    if path.endswith('.npz'):
        stats = np.load(path)
        m, s = stats['mu'], stats['sigma']
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        dataset = FilesDataset(files)
        m, s = compute_statistics_for_dataset(dataset, model, batch_size, device)

    return m, s


def load_model(dims=2048) -> torch.nn.Module:
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])

    return model


def compute_fid_given_paths(path_real, path_fake, batch_size: int, dims, device: str='cpu'):
    """Calculates the FID of two paths"""
    model = load_model(dims).to(device)
    m1, s1 = compute_statistics_for_path(path_real, model, batch_size, device)
    m2, s2 = compute_statistics_for_path(path_fake, model, batch_size, device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def read_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')

    # `compute_fid` command
    subparser = subparsers.add_parser('compute_fid')
    subparser.add_argument('--path_real', type=str, help=('Path to the real images or to .npz statistic files'))
    subparser.add_argument('--path_fake', type=str, help=('Path to the generated images or to .npz statistic files'))
    subparser.add_argument('--batch-size', type=int, default=50, help='Batch size to use')
    subparser.add_argument('--dims', type=int, default=2048, choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. By default, uses pool3 features'))

    # `compute_stats` command
    subparser = subparsers.add_parser('compute_stats')
    subparser.add_argument('--data_path', type=str, nargs=1, help='Path to the images')
    subparser.add_argument('--save_path', type=str, nargs=1, help='Path where to save the .npz statistic files')
    subparser.add_argument('--batch-size', type=int, default=50, help='Batch size to use')
    subparser.add_argument('--dims', type=int, default=2048, choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. By default, uses pool3 features'))

    return parser.parse_args()


if __name__ == '__main__':
    args = read_args()

    if torch.cuda.is_available():
        print('Going to use GPU')
        device = 'gpu'
    else:
        device = 'cpu'

    if args.command == 'compute_fid':
        fid_value = compute_fid_given_paths(args.path_real, args.path_fake, args.batch_size, device)
        print('FID: ', fid_value)
    elif args.command == 'compute_stats':
        model = load_model(args.dims).to(args.device)
        m, s = compute_statistics_for_path(args.data_path, model, args.batch_size)
        np.save(args.save_path, mu=mu, sigma=s)
    else:
        raise ValueError(f'Unknown command: {args.command}')
