import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import autograd

from dinr.utils.comm import AllReduce, get_world_size


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, old_running_path_mean, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
    path_mean = path_lengths.mean()
    if get_world_size() > 1:
        path_mean_reduced = AllReduce.apply(path_mean) * (1.0 / dist.get_world_size())
    else:
        path_mean_reduced = path_mean
    running_path_mean = old_running_path_mean + decay * (path_mean_reduced - old_running_path_mean)
    path_penalty = (path_lengths - running_path_mean).pow(2).mean()

    return path_penalty, running_path_mean.detach(), path_mean_reduced.detach()
