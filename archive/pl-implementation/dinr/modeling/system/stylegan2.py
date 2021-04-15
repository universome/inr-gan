from typing import Optional, List, Tuple, Dict, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.utils.data import Dataset
from tqdm import tqdm

from ..metrics.fid import FID
from ..metrics.inception import InceptionV3Wrapper
from ..metrics.inception_score import InceptionScore
from ..modules.loss import d_logistic_loss, d_r1_loss, g_nonsaturating_loss, g_path_regularize
from ..modules.noise import mixing_noise
from ...data.build import build_datasets, build_loaders
from ...utils.callbacks import EMAWeightUpdate
from ...utils.comm import get_world_size, is_main_process
from ...utils.hydra import instantiate


class StyleGAN2(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.hparams = cfg

        self.generator = instantiate(self.hparams.model.generator)
        self.generator_ema = instantiate(self.hparams.model.generator)
        for param in self.generator_ema.parameters():
            param.requires_grad = False

        # self.datasets: Dict[stage, Optional[List[Tuple[dataset_name, dataset_config, dataset_class]]]]
        # defined when calling {train/val/test}_dataloader
        self.datasets: Dict[str, Optional[List[Tuple[str, DictConfig, Dataset]]]] = \
            {'train': None, 'val': None, 'test': None}

        # Copy weights from generator to generator_ema
        EMAWeightUpdate.update_weights(self.generator, self.generator_ema, 0.0)

        # Define FID and InceptionScore metrics

        # metric: Dict[stage, [metric_for_dataset_i,...]]
        self.fid: Dict[str, nn.ModuleList] = {}
        self.inception_score: Dict[str, nn.ModuleList] = {}

        self._inception = InceptionV3Wrapper()

        if not self.hparams.eval_only:
            self.discriminator = instantiate(self.hparams.model.discriminator)
            self.fid['val'] = nn.ModuleList([FID(self.hparams.datasets.val[i].inception_real_stats_path) for i in
                                             range(len(self.hparams.datasets.val))])
            self.inception_score['val'] = nn.ModuleList(
                [InceptionScore() for _ in range(len(self.hparams.datasets.val))])
        else:
            self.fid['test'] = nn.ModuleList([FID(self.hparams.datasets.test[i].inception_real_stats_path) for i in
                                              range(len(self.hparams.datasets.test))])
            self.inception_score['test'] = nn.ModuleList(
                [InceptionScore() for _ in range(len(self.hparams.datasets.test))])

        self.fid = nn.ModuleDict(self.fid)
        self.inception_score = nn.ModuleDict(self.inception_score)

        self.register_buffer("mean_path_length", torch.tensor(0))

        # TODO(universome): replace with diffaugs
        if 'random_aug' in self.hparams.model:
            print('Initialized random augs!')
            self.random_aug = instantiate(self.hparams.model.random_aug)
        else:
            self.random_aug = None

    def forward(self, z, return_latents=False, **generator_forward_kwargs):
        if self.training:
            fake_img, latent = self.generator(z, return_latents=return_latents, **generator_forward_kwargs)
        else:
            fake_img, latent = self.generator_ema(z, return_latents=return_latents, **generator_forward_kwargs)
            fake_img = fake_img.clamp(-1.0, 1.0)

        if return_latents:
            return fake_img, latent

        return fake_img

    def on_train_start(self):
        num_samples_for_fid = self.hparams.fid.num_samples
        batch_size_for_fid = self.hparams.fid.batch_size

        assert num_samples_for_fid % get_world_size() == 0
        assert (num_samples_for_fid // get_world_size()) % batch_size_for_fid == 0

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_img = batch['img']
        batch_size = real_img.size(0)

        if optimizer_idx == 0:
            noise = mixing_noise(batch_size, self.hparams.system.latent, self.hparams.system.mixing, self.device)
            fake_img = self.forward(noise)

            if not self.random_aug is None:
                fake_img = self.random_aug(fake_img)
                real_img = self.random_aug(real_img)

            fake_pred = self.discriminator(fake_img.detach())
            real_pred = self.discriminator(real_img)
            d_loss = d_logistic_loss(real_pred, fake_pred)
            self.log("d", d_loss, prog_bar=True, logger=True, sync_dist=True)
            self.log("real_score", real_pred.mean(), prog_bar=True, logger=True, sync_dist=True)
            self.log("fake_score", fake_pred.mean(), prog_bar=True, logger=True, sync_dist=True)

            loss = d_loss
        elif optimizer_idx == 1:
            real_img.requires_grad = True

            if not self.random_aug is None:
                real_img = self.random_aug(real_img)

            real_pred = self.discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)
            weighted_r1_loss = self.hparams.system.r1 / 2 * r1_loss * self.hparams.system.d_reg_every + 0 * real_pred[0]
            self.log("r1", r1_loss, prog_bar=True, logger=True, sync_dist=True)

            loss = weighted_r1_loss
        elif optimizer_idx == 2:
            noise = mixing_noise(batch_size, self.hparams.system.latent, self.hparams.system.mixing, self.device)
            fake_img = self.forward(noise)

            if not self.random_aug is None:
                fake_img = self.random_aug(fake_img)

            fake_pred = self.discriminator(fake_img)
            g_loss = g_nonsaturating_loss(fake_pred)
            self.log("g_loss", g_loss, prog_bar=True, logger=True, sync_dist=True)

            loss = g_loss
        elif optimizer_idx == 3:
            path_batch_size = max(1, batch_size // self.hparams.system.path_batch_shrink)
            noise = mixing_noise(path_batch_size, self.hparams.system.latent, self.hparams.system.mixing, self.device)
            fake_img, latents = self.forward(noise, return_latents=True)

            if not self.random_aug is None:
                fake_img = self.random_aug(fake_img)

            path_loss, self.mean_path_length, path_mean = g_path_regularize(fake_img, latents, self.mean_path_length)

            weighted_path_loss = self.hparams.system.path_regularize * self.hparams.system.g_reg_every * path_loss

            if self.hparams.system.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            self.log("path", path_loss, prog_bar=True, logger=True, sync_dist=True)
            self.log("path_length", path_mean, prog_bar=True, logger=True, sync_dist=False)

            loss = weighted_path_loss
        else:
            raise NotImplementedError(f'Unknown optimizer idx: {optimizer_idx}')

        return loss

    def optimizer_step(
            self,
            *args,
            epoch: int = None,
            batch_idx: int = None,
            optimizer: Optimizer = None,
            optimizer_idx: int = None,
            optimizer_closure: Optional[Callable] = None,
            on_tpu: bool = None,
            using_native_amp: bool = None,
            using_lbfgs: bool = None,
            **kwargs,
    ) -> None:
        if optimizer_idx == 0 or optimizer_idx == 2 or (optimizer_idx == 1 and self.d_regularize) or (
                optimizer_idx == 3 and self.g_regularize):
            optimizer_closure()
            optimizer.step()
            optimizer.zero_grad()

    @property
    def d_regularize(self):
        return self.hparams.system.d_reg_every > 0 and self.global_step % self.hparams.system.d_reg_every == 0

    @property
    def g_regularize(self):
        return self.hparams.system.g_reg_every > 0 and self.global_step % self.hparams.system.g_reg_every == 0

    def _shared_eval_step(self, batch, batch_idx: int, dataloader_idx: Optional[int], partition: str):
        fid_metric: FID = self.fid[partition][dataloader_idx]

        if fid_metric.real_stats is None:
            self._inception.eval()
            with torch.no_grad():
                inception_feats, _ = self._inception((batch['img'] + 1.0) / 2.0)
            fid_metric.update(inception_feats, images_are_real=True)

    def _shared_eval_epoch_end(self, _, partition: str):
        self.compute_fid_and_inception_for_partition(partition)

    @torch.no_grad()
    def compute_fid_and_inception_for_partition(self, partition: str):

        # Sample fake images
        num_samples = self.hparams.fid.num_samples
        batch_size = self.hparams.fid.batch_size
        assert num_samples % get_world_size() == 0
        assert (num_samples // get_world_size()) % batch_size == 0
        num_batch = (num_samples // get_world_size()) // batch_size
        batch_iterator = range(num_batch)
        self._inception.eval()

        if is_main_process():
            batch_iterator = tqdm(batch_iterator, desc="Generating samples for FID/IS")

        if self.hparams.fid.truncation < 1.0:
            noise = mixing_noise(self.hparams.fid.num_latents_for_trunc_proto, self.hparams.system.latent,
                                 prob=0.0, device=self.device)  # [num_latents_for_proto, z_dim]
            latents = self.generator.style(noise[0])  # [num_latents_for_trunc_proto, style_dim]
            latent_prototype = latents.mean(dim=0)  # [style_dim]
        else:
            latent_prototype = None

        for k in batch_iterator:
            noise = mixing_noise(batch_size, self.hparams.system.latent, self.hparams.system.mixing, self.device)
            fake_img = self.forward(
                noise,
                truncation=self.hparams.fid.truncation,
                truncation_latent=latent_prototype,
            )
            fake_img = (fake_img + 1.0) / 2.0  # [-1, 1] => [0, 1] range
            inception_feats, inception_logits = self._inception(fake_img)

            for i, (dataset_name, _, _) in enumerate(self.datasets[partition]):
                self.fid[partition][i].update(inception_feats, images_are_real=False)
                self.inception_score[partition][i].update(inception_logits)

        # Report FID and IS for each val or test dataset
        for i, (dataset_name, _, _) in enumerate(self.datasets[partition]):
            # Workaround because pl is still not ideal (maybe it would work with lists not tensors)
            self.fid[partition][i].real_activations = self.fid[partition][i].real_activations.to(self.device)
            self.fid[partition][i].fake_activations = self.fid[partition][i].fake_activations.to(self.device)
            self.inception_score[partition][i].fake_logits = self.inception_score[partition][i].fake_logits.to(
                self.device)

            self.log(f"FID/{dataset_name}", self.fid[partition][i].compute(), prog_bar=True, logger=True)
            self.log(f"IS/{dataset_name}", self.inception_score[partition][i].compute(), prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx: int, *args):
        dataloader_idx = args[0] if len(args) == 1 else 0
        return self._shared_eval_step(batch, batch_idx, dataloader_idx, 'val')

    def test_step(self, batch, batch_idx: int, *args):
        dataloader_idx = args[0] if len(args) == 1 else 0
        return self._shared_eval_step(batch, batch_idx, dataloader_idx, 'test')

    def validation_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, 'test')

    def configure_optimizers(self):
        assert self.hparams.optim.generator.optimizer['_target_'] == "torch.optim.Adam"
        assert self.hparams.optim.discriminator.optimizer['_target_'] == "torch.optim.Adam"

        lr_g = self.hparams.optim.generator.optimizer.lr
        lr_d = self.hparams.optim.discriminator.optimizer.lr
        betas_g = self.hparams.optim.generator.optimizer.betas
        betas_d = self.hparams.optim.discriminator.optimizer.betas

        g_reg_ratio = 1.0
        if self.hparams.system.g_reg_every > 0:
            g_reg_ratio = self.hparams.system.g_reg_every / (self.hparams.system.g_reg_every + 1)

        d_reg_ratio = 1.0
        if self.hparams.system.d_reg_every > 0:
            d_reg_ratio = self.hparams.system.d_reg_every / (self.hparams.system.d_reg_every + 1)

        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=lr_g * g_reg_ratio,
                                 betas=(betas_g[0] ** g_reg_ratio, betas_g[1] ** g_reg_ratio))
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=lr_d * d_reg_ratio,
                                 betas=(betas_d[0] ** d_reg_ratio, betas_d[1] ** d_reg_ratio))
        return [opt_d, opt_d, opt_g, opt_g], []

    def train_dataloader(self):
        self.datasets['train'] = build_datasets(self.hparams, 'train')
        assert len(self.datasets['train']) == 1, \
            "Multiple train datasets should be defined using the single dataset class e.g. ConcatDataset"
        loaders = build_loaders(self.hparams, self.datasets['train'], 'train')
        return loaders[0]

    def val_dataloader(self):
        self.datasets['val'] = build_datasets(self.hparams, 'val')
        return build_loaders(self.hparams, self.datasets['val'], 'val')

    def test_dataloader(self):
        self.datasets['test'] = build_datasets(self.hparams, 'test')
        return build_loaders(self.hparams, self.datasets['test'], 'test')

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)

        return items
