import os
import time
import random
from typing import Iterable, Callable, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from firelab.base_trainer import BaseTrainer
from firelab.config import Config
from firelab.utils.data_utils import text_to_markdown
from firelab.utils.training_utils import PiecewiseLinearScheme
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import horovod.torch as hvd
import torch.multiprocessing as mp

from src.models.gan import GAN32
from src.models.resnet_gan import ResnetGAN
from src.models.inr_gan import INRGAN
from src.utils.training_utils import construct_optimizer, compute_covariance
from src.utils.losses import compute_r1_penalty_from_outputs, compute_wgan_gp
from src.dataloaders.load_data import load_data, CenterCropToMin
from src.dataloaders.utils import ProgressiveTransform
from src.utils.fid.compute_fid import compute_statistics_for_dataset, load_model, calculate_frechet_distance_torch
from src.utils.constants import DEBUG


class GANTrainer(BaseTrainer):
    def __init__(self, config: Config):
        config = config.overwrite(config.datasets[config.dataset]) # Overwriting with the dataset-dependent hyperparams
        config = config.overwrite(Config.read_from_cli()) # Overwriting with the CLI arguments
        config = config.overwrite(Config({'datasets': None})) # So not to pollute logs

        super(GANTrainer, self).__init__(config)

        if self.is_distributed:
            torch.set_num_threads(4)

    def build_model(self) -> nn.Module:
        if self.config.model_type == 'gan32':
            return GAN32(self.config)
        elif self.config.model_type == 'resnet_gan':
            return ResnetGAN(self.config)
        elif self.config.model_type == 'inr_gan':
            return INRGAN(self.config)
        else:
            raise NotImplementedError(f'Unknown model type: {self.config.model_type}')

    def is_main_process(self):
        return (not self.is_distributed) or (hvd.rank() == 0)

    def init_models(self):
        self.model = self.build_model().to(self.device_name)

        if self.is_main_process():
            if self.config.model_type in {'gan32', 'resnet_gan'}:
                self.logger.info(f'Number of parameters in Generator: {sum(p.numel() for p in self.model.generator.parameters())}')
                self.logger.info(f'Number of parameters in Discriminator: {sum(p.numel() for p in self.model.discriminator.parameters())}')
            elif self.config.model_type == 'inr_gan':
                self.logger.info(f'Number of parameters in Generator: {sum(p.numel() for p in self.model.generator.parameters())}')
                self.logger.info(f'Generator output projection matrix size: {self.model.generator.get_output_matrix_size()}')
                # self.logger.info(f'Num learnable parameters in INR: {sum(p.numel() for p in self.model.generator.inr.parameters() if p.requires_grad)}')
                self.logger.info(f'Number of parameters in Discriminator: {sum(p.numel() for p in self.model.discriminator.parameters())}')
                self.logger.info(f'INR size (generator output dim): {self.model.generator.inr.num_external_params}')
            else:
                raise NotImplementedError

        if self.config.model_type == 'inr_gan':
            if self.config.hp.inr.get('has_actnorm', False):
                # Initializating actnorm parameters
                with torch.no_grad():
                    self.model.generator.generate_image(512, self.device_name)

        if self.config.hp.get('gen_ema.enabled'):
            self.gen_ema = self.build_model().generator.to(self.device_name)
            self.gen_ema.load_state_dict(self.model.generator.state_dict())
            self.gen_ema.eval()

        if self.config.hp.get('fid_loss.enabled'):
            self.feat_extractor = load_model(2048).to(self.device_name)
            m_real, cov_real = self.compute_inception_stats_real(self.feat_extractor, 8192, self.config.data.target_img_size)

            self.mean_real = torch.from_numpy(m_real).float().to(self.device_name)
            self.cov_real = torch.from_numpy(cov_real).float().to(self.device_name)

        torch.cuda.set_device(self.device_name)

    def update_gen_ema(self):
        if not self.config.hp.get('gen_ema.enabled'): return

        for p_src, p_tgt in zip(self.model.generator.parameters(), self.gen_ema.parameters()):
            assert(p_src is not p_tgt)
            p_tgt.data.copy_(self.config.hp.gen_ema.ema_coef * p_tgt.data + (1. - self.config.hp.gen_ema.ema_coef) * p_src.data)

    def init_dataloaders(self):
        self.dataset = load_data(self.config.data)
        kwargs = {
            'batch_size': self.config.hp.batch_size,
            'drop_last': True,
            'num_workers': 4,
            'pin_memory': True
        }
        if self.is_distributed:
            kwargs['multiprocessing_context'] = 'forkserver'
            self.data_sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset, num_replicas=hvd.size(), rank=hvd.rank())
        else:
            self.data_sampler = None
        self.train_dataloader = torch.utils.data.DataLoader(self.dataset, sampler=self.data_sampler, **kwargs)

    def init_optimizers(self):
        self.discriminator_optim = construct_optimizer(self.model.discriminator, self.config.hp.discr_optim)
        self.generator_optim = construct_optimizer(self.model.generator, self.config.hp.gen_optim)

        if self.is_distributed:
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            hvd.broadcast_parameters(self.gen_ema.state_dict(), root_rank=0)

            self.discriminator_optim = hvd.DistributedOptimizer(self.discriminator_optim,
                named_parameters=[(f'discriminator.{k}', p) for k, p in self.model.discriminator.named_parameters()],
                **self.config.distributed_training.dist_optim_kwargs,
                backward_passes_per_step=self.config.hp.grad_accum_steps)
            self.generator_optim = hvd.DistributedOptimizer(self.generator_optim,
                named_parameters=[(f'generator.{k}', p) for k, p in self.model.generator.named_parameters()],
                **self.config.distributed_training.dist_optim_kwargs,
                backward_passes_per_step=self.config.hp.grad_accum_steps)

            hvd.broadcast_optimizer_state(self.discriminator_optim, root_rank=0)
            hvd.broadcast_optimizer_state(self.generator_optim, root_rank=0)

        if self.config.hp.grad_penalty.get('schedule.enabled'):
            self.gp_schedule = PiecewiseLinearScheme(
                self.config.hp.grad_penalty.schedule.values,
                self.config.hp.grad_penalty.schedule.iters,
            )

    def generate_fixed_generator_inputs(self, size: int) -> Tensor:
        inputs = self.model.generator.sample_noise(size, self.config.hp.test_time_noise_correction).to(self.device_name)

        if self.config.data.is_conditional:
            labels = torch.randint(0, self.config.data.num_classes, size=(size,)).to(self.device_name) # [size]
            class_embs = self.model.generator.class_embedder(labels) # [size, class_emb_dim]
            inputs = torch.cat([inputs, class_embs], dim=1) # [size, z_dim + class_emb_dim]

        if self.config.data.is_variable_sized:
            self.aspect_ratios = self.model.generator.size_sampler.sample_aspect_ratios(labels) # [size, 1]
            aspect_ratios_embs = self.model.generator.size_sampler.pos_embedder(self.aspect_ratios.unsqueeze(1)) # [size, aspect_ratio_emb_dim]
            inputs = inputs = torch.cat([inputs, aspect_ratios_embs], dim=1) # [size, z_dim + class_emb_dim + aspect_ratio_emb_dim]

        self.fixed_noise = inputs

    def after_init_hook(self):
        # Fixed noise for validation
        assert self.config.fid.num_fake_images % self.config.fid.batch_size == 0, \
            "Choose a batch-size divisible number for `num_fake_images`"

        self.generate_fixed_generator_inputs(self.config.fid.num_fake_images)

        if self.is_main_process():
            self.writer.add_text('Config', self.config.to_markdown(), self.num_iters_done)
            self.writer.add_text('Generator', text_to_markdown(str(self.model.generator)), self.num_iters_done)
            self.writer.add_text('Discriminator', text_to_markdown(str(self.model.discriminator)), self.num_iters_done)

            if self.config.model_type == 'inr_gan':
                self.writer.add_text('INR', text_to_markdown(str(self.model.generator.inr)), self.num_iters_done)

        self.setup_scheduler()

    def setup_scheduler(self):
        if not self.config.hp.get('cyclic_scheduler.enabled'):
            return

        gen_optim_groups = sorted(self.config.hp.cyclic_scheduler.gen_optim.keys())
        gen_base_lrs = [self.config.hp.cyclic_scheduler.gen_optim[g][0] for g in gen_optim_groups]
        gen_max_lrs = [self.config.hp.cyclic_scheduler.gen_optim[g][1] for g in gen_optim_groups]

        self.gen_scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.generator_optim, base_lr=gen_base_lrs, max_lr=gen_max_lrs,
            step_size_up=2500, cycle_momentum=False)
        self.discr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.discriminator_optim, base_lr=self.config.hp.cyclic_scheduler.discr_optim[0],
            max_lr=self.config.hp.cyclic_scheduler.discr_optim[1],
            step_size_up=2500, cycle_momentum=False)

        # Now, since pytorch has a bug in setting last_epoch, we have to loop manually...
        for _ in range(self.num_iters_done): self.gen_scheduler.step()
        for _ in range(self.num_iters_done + 2500): self.discr_scheduler.step()

    def on_epoch_done(self):
        if self.is_distributed:
            self.data_sampler.set_epoch(self.num_epochs_done)

    def train_on_batch(self, batch):
        self.model.train()

        x = batch[0].to(self.device_name)

        if self.config.data.is_conditional:
            y = batch[1].to(self.device_name)
        else:
            y = None

        self.run_fn_with_profiling(self.discriminator_step, x, y)

        if self.num_iters_done % self.config.hp.num_discr_steps_per_gen_step == 0:
            self.run_fn_with_profiling(self.generator_step)

        if self.config.hp.get('progressive_transform.enabled'):
            self.transform.update(self.num_iters_done)
            self.model.generator.inr.min_scale = self.transform.curr_min_scale
            if self.is_main_process():
                self.writer.add_scalar('progressive_tansform/min_scale', self.transform.curr_min_scale, self.num_iters_done)

        if self.config.hp.get('cyclic_scheduler.enabled'):
            self.gen_scheduler.step()
            self.discr_scheduler.step()

            if self.is_main_process():
                for g_name, g in zip(self.config.hp.cyclic_scheduler.gen_optim.keys(), self.generator_optim.param_groups):
                    self.writer.add_scalar(f'lrs/gen_{g_name}', g['lr'], self.num_iters_done)
                self.writer.add_scalar('lrs/discr', self.discriminator_optim.param_groups[0]['lr'], self.num_iters_done)

        self.perform_logging()

    def perform_logging(self):
        # TODO(universome): this should be implemented with callbacks I think
        if not self.is_main_process(): return

        if self.check_if_time_to_log('activations'):
            self.run_fn_with_profiling(self.log_activations)

        if self.check_if_time_to_log('weights'):
            self.run_fn_with_profiling(self.log_weights)

        # We log grads in .perform_optim_step() to make things faster
        # if self.check_if_time_to_log('grads'):
        #     self.log_grads()

        if self.check_if_time_to_log('images'):
            self.run_fn_with_profiling(self.log_images)

        if self.check_if_time_to_log('upsampled_images', ignore_first_iter=True):
            self.run_fn_with_profiling(self.log_upsampled_images)

        if self.check_if_time_to_log('interpolations', ignore_first_iter=True):
            self.run_fn_with_profiling(self.log_interpolations)

        if self.check_if_time_to_log('weights_interpolations', ignore_first_iter=True):
            self.run_fn_with_profiling(self.log_weights_interpolations)

        if self.check_if_time_to_log('fid', ignore_first_iter=True):
            self.run_fn_with_profiling(self.compute_fid)

        if self.config.model_type == 'inr_gan' and self.check_if_time_to_log('fid_upsampled', ignore_first_iter=True) and not self.config.data.is_variable_sized:
            self.run_fn_with_profiling(self.compute_upsampled_fid)

    def check_if_time_to_log(self, property_name: str, ignore_first_iter: bool=False) -> bool:
        if ignore_first_iter and self.num_iters_done == 0:
            return False

        if self.num_iters_done % self.config.hp.grad_accum_steps != 0:
            return False

        num_full_iters_done = self.num_iters_done // self.config.hp.grad_accum_steps

        is_switched_on = self.config.logging.freqs.get(property_name, -1) > 0
        is_initial_freq_log_enabled = self.config.logging.get(f'initial_freq_log_iter.{property_name}', float('-inf')) > num_full_iters_done
        is_right_iter = num_full_iters_done % self.config.logging.freqs.get(property_name) == 0

        return is_switched_on and (is_right_iter or is_initial_freq_log_enabled)

    def should_log_generations_now(self):
        is_switched_on = self.config.logging.get('activations_log_freq', -1) > 0
        is_right_iter = self.num_iters_done % self.config.logging.activations_log_freq == 0

    def run_fn_with_profiling(self, fn: Callable, *args, _log_name: str=None, **kwargs):
        start_time = time.time()
        fn(*args, **kwargs)
        _log_name = _log_name if not _log_name is None else fn.__name__

        if self.is_main_process():
            self.writer.add_scalar(f'timings/{_log_name}', time.time() - start_time, self.num_iters_done)

    @torch.no_grad()
    def log_activations(self):
        x_fake, gen_activations, inr_activations = self.model.generator.generate_image(
            self.config.hp.batch_size, self.device_name, return_activations=True)

        for model_name, activations in [('gen', gen_activations), ('inr', inr_activations)]:
            for act_log_name, activations_values in activations.items():
                values_to_log = self.pick_k_random_values(activations_values.view(-1))
                self.writer.add_histogram(f'{model_name}_activations/{act_log_name}', values_to_log, self.num_iters_done)
                self.writer.add_scalar(f'{model_name}_activations_mean/{act_log_name}', activations_values.mean(), self.num_iters_done)
                self.writer.add_scalar(f'{model_name}_activations_std/{act_log_name}', activations_values.std(), self.num_iters_done)

    @torch.no_grad()
    def log_weights(self):
        self.log_module_data(self.model.generator.named_parameters(), prefix='generator_weights')
        self.log_module_data(self.model.discriminator.named_parameters(), prefix='discriminator_weights')

    @torch.no_grad()
    def log_grads(self, module_name: str):
        grads = [(p_name, p.grad) for (p_name, p) in getattr(self.model, module_name).named_parameters()]
        self.log_module_data(grads, prefix=f'{module_name}_grads')

    @torch.no_grad()
    def log_module_data(self, data: Iterable, prefix: str):
        for key, value in data:
            values_to_log = self.pick_k_random_values(value.data.cpu().view(-1))
            self.writer.add_histogram(f'{prefix}/{key}', values_to_log, self.num_iters_done)
            self.writer.add_scalar(f'{prefix}_mean/{key}', value.mean().item(), self.num_iters_done)
            self.writer.add_scalar(f'{prefix}_std/{key}', value.std().item(), self.num_iters_done)
            self.writer.add_scalar(f'{prefix}_norm/{key}', value.norm().item(), self.num_iters_done)

    def pick_k_random_values(self, values: Tensor) -> Tensor:
        k = self.config.logging.num_hist_values_to_log
        values = values.view(-1)

        if len(values) >= k:
            return values[random.sample(range(len(values)), k)]
        else:
            return values

    def discriminator_step(self, x_real: Tensor, labels_real: Tensor=None):
        with torch.no_grad():
            x_fake, labels_fake = self.model.generator.generate_image(self.config.hp.batch_size, self.device_name, return_labels=True)

        if self.config.hp.grad_penalty.type == "r1":
            x_real.requires_grad_(True)

        adv_logits_on_real = self.model.discriminator(x_real, labels_real)
        adv_logits_on_fake = self.model.discriminator(x_fake, labels_fake)

        if self.config.hp.grad_penalty.get('schedule.enabled'):
            gp_weight = self.gp_schedule.evaluate(self.num_iters_done)
        else:
            gp_weight = self.config.hp.grad_penalty.weight

        if gp_weight > 0.0:
            if self.config.hp.grad_penalty.type == "r1":
                grad_penalty = compute_r1_penalty_from_outputs(adv_logits_on_real, x_real)
            elif self.config.hp.grad_penalty.type == "wgan-gp":
                grad_penalty = compute_wgan_gp(discriminator, x_real, x_fake)
            elif self.config.hp.grad_penalty.type in {"none", None}:
                grad_penalty = torch.tensor([0.0]).to(self.device_name)
            else:
                raise NotImplementedError(f'Unknown gradient penalty: {self.config.hp.grad_penalty.type}')

            if self.is_main_process():
                self.writer.add_scalar('discr/penalty', grad_penalty.item(), self.num_iters_done)
        else:
            grad_penalty = torch.tensor([0.0]).to(self.device_name)

        adv_loss_real = self.compute_loss(adv_logits_on_real, True)
        adv_loss_fake = self.compute_loss(adv_logits_on_fake, False)
        adv_loss = adv_loss_real + adv_loss_fake
        total_loss = adv_loss + gp_weight * grad_penalty

        self.perform_optim_step(total_loss, 'discriminator')

        if self.is_main_process():
            self.writer.add_scalar('discr/adv_loss', adv_loss.item(), self.num_iters_done)
            self.writer.add_scalar('discr/mean_logits/real', adv_logits_on_real.mean().item(), self.num_iters_done)
            self.writer.add_scalar('discr/mean_logits/fake', adv_logits_on_fake.mean().item(), self.num_iters_done)

            if self.config.hp.gan_loss_type == "standard":
                self.writer.add_scalar('discr/accuracy/real', (adv_logits_on_real > 0).float().mean().item(), self.num_iters_done)
                self.writer.add_scalar('discr/accuracy/fake', (adv_logits_on_fake < 0).float().mean().item(), self.num_iters_done)

            if self.config.hp.grad_penalty.get('schedule.enabled'):
                self.writer.add_scalar('discr/gp_weight', gp_weight, self.num_iters_done)

    def generator_step(self):
        total_loss = 0.0

        x_fake, labels_fake = self.model.generator.generate_image(self.config.hp.batch_size, self.device_name, return_labels=True)

        self.toggle_discr_grads(False) # TODO: does it break discr grads?
        adv_logits_on_fake = self.model.discriminator(x_fake, labels_fake)
        self.toggle_discr_grads(True)
        adv_loss = self.compute_loss(adv_logits_on_fake, True)

        if self.config.hp.get('fid_loss.enabled'):
            feats_fake = self.feat_extractor(x_fake * 0.5 + 0.5)[0] # [-1, 1] => [0, 1]
            feats_fake = feats_fake.view(len(feats_fake), -1)
            mean_fake = feats_fake.mean(dim=0)
            cov_fake = compute_covariance(feats_fake)
            fid = calculate_frechet_distance_torch(mean_fake, cov_fake, self.mean_real, self.cov_real,
                num_approx_iters=self.config.hp.fid_loss.num_approx_iters)
            fid_loss = self.config.hp.fid_loss.loss_coef * fid
            total_loss += fid_loss

            if self.is_main_process():
                self.writer.add_scalar(f'gen/batch_fid', fid.item(), self.num_iters_done)

        total_loss += adv_loss

        self.perform_optim_step(total_loss, 'generator')

        if self.is_main_process():
            self.writer.add_scalar(f'gen/adv_loss', adv_loss.item(), self.num_iters_done)

        self.update_gen_ema()

    def toggle_discr_grads(self, flag: bool):
        """
        We need to toggle the grads so horovod does not complain.
        For a single-gpu training this is not needed, but I have a suspicion
        that it would work a tiny bit faster if we do not ask
        pytorch to save grads for D's weights during G's forward.
        """
        for p in self.model.discriminator.parameters():
            p.requires_grad_(flag)

    def compute_loss(self, discr_logits: Tensor, is_real: bool) -> Tensor:
        if self.config.hp.gan_loss_type == "standard":
            targets = discr_logits.new_full(size=discr_logits.size(), fill_value=int(is_real))

            return F.binary_cross_entropy_with_logits(discr_logits, targets)
        elif self.config.hp.gan_loss_type == "wgan":
            return discr_logits.mean() * (1 if is_real else -1)
        else:
            raise NotImplementedError(f'Unknown gan_loss_type: {self.config.hp.gan_loss_type}')

    def perform_optim_step(self, loss, module_name: str, retain_graph: bool=False):
        optim = getattr(self, f'{module_name}_optim')

        loss /= self.config.hp.grad_accum_steps
        loss.backward(retain_graph=retain_graph)

        if self.is_distributed and self.num_iters_done % self.config.hp.grad_accum_steps == 0:
            # Syncing here since we can use it in logging and grad clipping
            optim.synchronize()

        if self.check_if_time_to_log('grads') and self.is_main_process():
            self.run_fn_with_profiling(self.log_grads, module_name, _log_name=f'log_grads_{module_name}')

        if self.config.hp.get(f'grad_clipping.{module_name}', -1) > 0:
            grad_clip_val = self.config.hp.grad_clipping.get(module_name)
            grad_norm = nn.utils.clip_grad_norm_(getattr(self.model, module_name).parameters(), grad_clip_val)
            self.writer.add_scalar(f'grad_norms/{module_name}', grad_norm, self.num_iters_done)

        if self.num_iters_done % self.config.hp.grad_accum_steps == 0:
            if self.is_distributed:
                with optim.skip_synchronize():
                    optim.step()
            else:
                optim.step()

            optim.zero_grad()
        else:
            # Gradient accumulation is enabled and it's not time to update
            pass

    @torch.no_grad()
    def sample_grid(self, img_size: int):
        fixed_noise = self.fixed_noise[:self.config.logging.num_imgs_to_display]
        batch_size, z_dim = self.config.logging.log_imgs_batch_size, fixed_noise.size(1)
        n_iters = len(fixed_noise) // batch_size
        inputs = fixed_noise.view(n_iters, batch_size, z_dim)

        if self.config.data.is_variable_sized:
            aspect_ratios = self.aspect_ratios[:n_iters * batch_size].view(n_iters, batch_size)
        else:
            aspect_ratios = [None] * n_iters

        imgs = torch.stack([self.gen_ema(zs, img_size, ars).cpu() for zs, ars in zip(inputs, aspect_ratios)])
        imgs = imgs.view(batch_size * n_iters, *imgs.shape[2:]) # [n_iters * batch_size, n_channels, img_size, img_size]
        imgs = imgs / 2 + 0.5
        samples_grid = make_grid(imgs, nrow=batch_size)

        return samples_grid, imgs

    @torch.no_grad()
    def log_images(self):
        self.gen_ema.eval()

        # Computing samples
        samples_grid, imgs = self.sample_grid(self.config.data.target_img_size)
        self.img_writer.add_image('Samples', samples_grid, self.num_iters_done)
        save_image(samples_grid, os.path.join(self.paths.custom_data_path, f'samples_iter_{self.num_iters_done}.png'))

    @torch.no_grad()
    def log_upsampled_images(self):
        upsample_factor = 2
        samples_grid, _ = self.sample_grid(self.config.data.target_img_size * upsample_factor)
        self.img_writer.add_image('inr_upsampled_images', samples_grid, self.num_iters_done)
        save_image(samples_grid, os.path.join(self.paths.custom_data_path, f'inr_upsampled_imgs_iter_{self.num_iters_done}.png'))

    @torch.no_grad()
    def log_interpolations(self):
        self.gen_ema.eval()

        # Computing interpolations
        fixed_noise = self.fixed_noise[:self.config.logging.num_imgs_to_display]
        batch_size, n_interpolations, z_dim = self.config.logging.log_imgs_batch_size, 10, fixed_noise.size(1)
        z_from = fixed_noise[:batch_size] # [batch_size, z_dim]
        z_to = fixed_noise[batch_size : batch_size * 2] # [batch_size, z_dim]
        alpha = torch.linspace(0, 1, n_interpolations).view(n_interpolations, 1, 1).to(self.device_name) # [n_interpolations, 1, 1]
        zs = z_from.view(1, batch_size, z_dim) * (1 - alpha) + z_to.view(1, batch_size, z_dim) * alpha # [n_interpolations, batch_size, z_dim]

        if self.config.data.is_variable_sized:
            alpha_ars = alpha.squeeze(2) # [n_interpolations, 1]
            aspect_ratios_from = self.aspect_ratios[:batch_size].unsqueeze(0) # [1, batch_size]
            aspect_ratios_to = self.aspect_ratios[batch_size : batch_size * 2].unsqueeze(0) # [1, batch_size]
            aspect_ratios = aspect_ratios_from * (1 - alpha_ars) + aspect_ratios_to * alpha_ars # [n_interpolations, batch_size]
        else:
            aspect_ratios = [None] * n_interpolations # [n_interpolations]

        imgs = torch.stack([self.gen_ema(curr_zs, aspect_ratios=ars).cpu() for curr_zs, ars in zip(zs, aspect_ratios)]) # [n_interpolations, batch_size, n_channels, img_size, img_size]
        imgs = imgs.permute(1, 0, 2, 3, 4) # [batch_size, n_interpolations, n_channels, img_size, img_size]
        imgs = imgs.reshape(n_interpolations * batch_size, *imgs.shape[2:]) # [batch_size * n_interpolations, n_channels, img_size, img_size]
        imgs = imgs / 2 + 0.5
        interpolations_grid = make_grid(imgs, nrow=n_interpolations)

        self.img_writer.add_image(f'Interpolations', interpolations_grid, self.num_iters_done)
        save_image(interpolations_grid, os.path.join(self.paths.custom_data_path, f'interpolations_iter_{self.num_iters_done}.png'))

    @torch.no_grad()
    def log_weights_interpolations(self):
        self.gen_ema.eval()

        # Computing interpolations
        fixed_noise = self.fixed_noise[:self.config.logging.num_imgs_to_display]
        batch_size, n_interpolations, z_dim = self.config.logging.log_imgs_batch_size, 10, fixed_noise.size(1)
        z_from = fixed_noise[:batch_size] # [batch_size, z_dim]
        z_to = fixed_noise[batch_size : batch_size * 2] # [batch_size, z_dim]

        # We are interpolating in INR space
        inr_ext_size = self.gen_ema.inr.num_external_params
        inr_weights_from = self.gen_ema.compute_model_forward(z_from).unsqueeze(0) # [1, batch_size, inr_ext_size]
        inr_weights_to = self.gen_ema.compute_model_forward(z_to).unsqueeze(0) # [1, batch_size, inr_ext_size]

        alpha = torch.linspace(0, 1, n_interpolations).view(n_interpolations, 1, 1).to(self.device_name) # [n_interpolations, 1, 1]
        ws = inr_weights_from * (1 - alpha) + inr_weights_to * alpha # [n_interpolations, batch_size, z_dim]

        if self.config.data.is_variable_sized:
            alpha_ars = alpha.squeeze(2) # [n_interpolations, 1]
            aspect_ratios_from = self.aspect_ratios[:batch_size].unsqueeze(0) # [1, batch_size]
            aspect_ratios_to = self.aspect_ratios[batch_size : batch_size * 2].unsqueeze(0) # [1, batch_size]
            aspect_ratios = aspect_ratios_from * (1 - alpha_ars) + aspect_ratios_to * alpha_ars # [n_interpolations, batch_size]
        else:
            aspect_ratios = [None] * n_interpolations # [n_interpolations]

        imgs = torch.stack([self.gen_ema.forward_for_weights(w, aspect_ratios=a).cpu() for w, a in zip(ws, aspect_ratios)]) # [n_interpolations, batch_size, n_channels, img_size, img_size]
        imgs = imgs.permute(1, 0, 2, 3, 4) # [batch_size, n_interpolations, n_channels, img_size, img_size]
        imgs = imgs.reshape(n_interpolations * batch_size, *imgs.shape[2:]) # [batch_size * n_interpolations, n_channels, img_size, img_size]
        imgs = imgs / 2 + 0.5
        interpolations_grid = make_grid(imgs, nrow=n_interpolations)

        self.img_writer.add_image(f'Weights_interpolations', interpolations_grid, self.num_iters_done)
        save_image(interpolations_grid, os.path.join(self.paths.custom_data_path, f'weights_interpolations_iter_{self.num_iters_done}.png'))

    @torch.no_grad()
    def compute_inception_stats_real(self, inception_model: nn.Module, num_images: int, img_size: int) -> Tuple[np.ndarray, np.ndarray]:
        stats_file_path = f'{self.paths.custom_data_path}/real_data_fid_stats_{img_size}.npz'

        if os.path.exists(stats_file_path):
            real_stats = np.load(stats_file_path)
            m_real, s_real = real_stats['mu'], real_stats['sigma']
        else:
            # We will have to compute the stats once
            if self.config.data.is_variable_sized:
                ds_real = load_data(self.config.data)
            else:
                ds_real = load_data(self.config.data, transform=transforms.Compose([
                    CenterCropToMin(),
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))

            if num_images < len(ds_real):
                ds_real = Subset(ds_real, list(range(0, num_images)))

            m_real, s_real = compute_statistics_for_dataset(
                ds_real, inception_model, self.config.fid.batch_size, self.device_name, is_ds_labeled=True)
            # Saving them for future use
            np.savez(stats_file_path, mu=m_real, sigma=s_real)

        return m_real, s_real

    @torch.no_grad()
    def compute_inception_stats_fake(self, inception_model: nn.Module, num_images: int, img_size: int=None) -> Tuple[np.ndarray, np.ndarray]:
        n_iters = num_images // self.config.fid.batch_size
        noise_ds = self.fixed_noise[:num_images].view(n_iters, self.config.fid.batch_size, -1).to(self.device_name)
        if self.config.data.is_variable_sized:
            aspect_ratios = self.aspect_ratios[:num_images].view(n_iters, self.config.fid.batch_size)
            ds_fake = torch.cat([self.gen_ema(zs, img_size=img_size, aspect_ratios=ars).cpu() for zs, ars in zip(noise_ds, aspect_ratios)], dim=0)
        else:
            ds_fake = torch.cat([self.gen_ema(zs, img_size=img_size).cpu() for zs in noise_ds], dim=0)

        ds_fake = ds_fake * 0.5 + 0.5

        m_fake, s_fake = compute_statistics_for_dataset(
            ds_fake, inception_model, self.config.fid.batch_size, self.device_name, is_ds_labeled=False)

        return m_fake, s_fake

    @torch.no_grad()
    def compute_fid(self):
        if DEBUG:
            dims, num_real_images, num_fake_images = 64, 128, 128
        else:
            dims, num_real_images, num_fake_images = self.config.fid.dims, self.config.fid.num_real_images, self.config.fid.num_fake_images

        self.gen_ema.eval()
        inception_model = load_model(dims).to(self.device_name)
        m_real, s_real = self.compute_inception_stats_real(inception_model, num_real_images, self.config.data.target_img_size)
        m_fake, s_fake = self.compute_inception_stats_fake(inception_model, num_fake_images)

        # Frechet Distance computation
        self.compute_and_log_fid(m_real, s_real, m_fake, s_fake, 'normal')

    @torch.no_grad()
    def compute_upsampled_fid(self):
        if DEBUG:
            dims, num_real_images, num_fake_images = 64, self.config.fid.batch_size * 4, self.config.fid.batch_size * 4
        else:
            dims, num_real_images, num_fake_images = self.config.fid.dims, self.config.fid.num_real_images, self.config.fid.num_fake_images

        self.gen_ema.eval()
        inception_model = load_model(dims).to(self.device_name)
        for scale_factor in self.config.logging.scale_factors:
            m_real, s_real = self.compute_inception_stats_real(
                inception_model, num_real_images, scale_factor * self.config.data.target_img_size)
            m_fake_inr_up, s_fake_inr_up = self.compute_inception_stats_fake(
                inception_model, num_fake_images, img_size=scale_factor * self.config.data.target_img_size)

            # Frechet Distance computation
            self.compute_and_log_fid(m_real, s_real, m_fake_inr_up, s_fake_inr_up, f'{scale_factor}x_upsampled')

    def compute_and_log_fid(self, m_real, s_real, m_fake, s_fake, prefix: str):
        if self.config.logging.get('decomposed_fid'):
            fid_vals = calculate_frechet_distance_torch(m_real, s_real, m_fake, s_fake, decomposed=True)

            for k in fid_vals:
                self.writer.add_scalar(f'FID/{prefix}_{k}', fid_vals[k], self.num_iters_done)
        else:
            fid = calculate_frechet_distance_torch(m_real, s_real, m_fake, s_fake)
            self.writer.add_scalar(f'FID/{prefix}', fid, self.num_iters_done)
