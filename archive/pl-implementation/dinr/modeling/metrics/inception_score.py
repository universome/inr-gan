import os
import logging

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import Metric
from scipy import linalg

log = logging.getLogger(__name__)


class InceptionScore(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        # Unfortunately, I still do not understand how many classes there should be
        # In "true" InceptionV3 for InceptionScore. For FID, it should be 1008
        self.add_state("fake_logits", default=torch.zeros(0, 0), dist_reduce_fx=None)

    def update(self, inception_logits: torch.Tensor):
        if self.fake_logits.numel() == 0:
            self.fake_logits = inception_logits.detach().cpu()
        else:
            self.fake_logits = torch.cat([self.fake_logits.cpu(), inception_logits.detach().cpu()], dim=0)

    def compute(self):
        probs = self.fake_logits.softmax(dim=1).cpu().numpy() # [num_samples, num_inception_classes]
        IS_mean, IS_std = calculate_inception_score(probs, num_splits=1)

        return IS_mean


def calculate_inception_score(pred, num_splits=10):
    """
    Calculates Inception Score mean + std given softmax'd logits and number of splits
    Copy-pasted from: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/inception_utils.py
    """
    scores = []
    for index in range(num_splits):
        pred_chunk = pred[index * (pred.shape[0] // num_splits): (index + 1) * (pred.shape[0] // num_splits), :]
        kl_inception = pred_chunk * (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
        kl_inception = np.mean(np.sum(kl_inception, 1))
        scores.append(np.exp(kl_inception))
    return np.mean(scores), np.std(scores)
