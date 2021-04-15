import logging
import math
import os
from datetime import timedelta
from time import perf_counter
from typing import List

import torch
import torchvision
from pytorch_lightning import Callback

from dinr.utils.comm import is_main_process

log = logging.getLogger(__name__)


class FitDurationCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        self.start_fit_time = perf_counter()

    def on_fit_end(self, trainer, pl_module):
        total_time = perf_counter() - self.start_fit_time
        log.info(f"Total trainer.fit() duration: {timedelta(seconds=int(total_time))}")


class CheckpointEveryNSteps(Callback):
    """
    Save a checkpoint every N steps
    """
    def __init__(self, save_step_frequency: int):
        """
        Args:
            save_step_frequency: how often to save in steps
        """
        self.save_step_frequency = save_step_frequency

    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.save_step_frequency == 0:
            filename = f"global_step={trainer.global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


class CheckpointAtImageNum(Callback):
    def __init__(self, image_nums: List[int], batch_size: int):
        self.step2imagenum = {}
        self.steps_list = []
        for image_num in image_nums:
            assert image_num % batch_size == 0
            step = image_num // batch_size
            self.steps_list.append(step)
            self.step2imagenum[step] = image_num

    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step in self.steps_list:
            filename = f"image_num={self.step2imagenum[trainer.global_step]}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


class GenerativeModelImageSampler(Callback):
    def __init__(self, num_samples: int, period_steps: int, nrow: int):
        super().__init__()
        self.num_samples = num_samples
        self.period_steps = period_steps
        self.nrow = nrow
        self.z = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        if trainer.global_step % self.period_steps == 0 and is_main_process():
            dim = (self.num_samples, pl_module.hparams.system.latent)

            if self.z is None:
                self.z = torch.normal(mean=0.0, std=1.0, size=dim, device=pl_module.device)

            # generate images
            with torch.no_grad():
                original_mode = pl_module.training
                pl_module.eval()
                images = []
                images_online = []
                for i in range(self.num_samples):
                    pl_module.eval()
                    images.append(pl_module(self.z[i].unsqueeze(0)))
                    pl_module.train()
                    images_online.append(pl_module(self.z[i].unsqueeze(0)))

                if self.num_samples > 1:
                    images = torch.cat(images, dim=0)
                    images_online = torch.cat(images_online, dim=0)
                else:
                    images = images[0]
                    images_online = images_online[0]
                pl_module.train(original_mode)

            grid = torchvision.utils.make_grid(images, self.nrow, normalize=True, range=(-1, 1))
            str_title = 'samples'
            trainer.logger[0].experiment.add_image(str_title, grid, global_step=trainer.global_step)

            images_online = images_online.clamp(-1.0, 1.0)
            grid = torchvision.utils.make_grid(images_online, self.nrow, normalize=True, range=(-1, 1))
            str_title = 'samples_online'
            trainer.logger[0].experiment.add_image(str_title, grid, global_step=trainer.global_step)


class GenerativeModelImageSamplerTest(Callback):
    def __init__(self, num_samples: int, nrow: int):
        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow

    def on_test_epoch_end(self, trainer, pl_module):
        if is_main_process():
            dim = (self.num_samples, pl_module.hparams.system.latent)

            z = torch.normal(mean=0.0, std=1.0, size=dim, device=pl_module.device)

            # generate images
            with torch.no_grad():
                original_mode = pl_module.training
                pl_module.eval()
                images = []
                for i in range(self.num_samples):
                    images.append(pl_module(z[i].unsqueeze(0)))
                if len(images) > 1:
                    images = torch.cat(images, dim=0)
                else:
                    images = images[0]
                pl_module.train(original_mode)

            grid = torchvision.utils.make_grid(images, self.nrow, normalize=True, range=(-1, 1))
            # str_title = f'{pl_module.__class__.__name__}_images'
            str_title = 'test_samples'
            trainer.logger[0].experiment.add_image(str_title, grid, global_step=trainer.global_step)


class PlotFourierSpectrum(Callback):
    def __init__(self, period_steps: int):
        super().__init__()
        self.period_steps = period_steps

        size = 256  # todo pass it using config
        x, y = torch.linspace(-1, 1, size, device='cpu'), torch.linspace(-1, 1, size, device='cpu')
        grid_y, grid_x = torch.meshgrid(y, x)
        grid_xy = torch.stack([grid_x, grid_y], dim=2).view(1, -1, 2)
        grid_xy = grid_xy.expand(1, grid_xy.size(1), 2)
        self.grid = grid_xy

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.global_step % self.period_steps == 0 and is_main_process() and hasattr(pl_module.generator,
                                                                                          'fourier_mapping'):
            with torch.no_grad():
                m = pl_module.generator.fourier_mapping
                argmax = (m.basis ** 2).sum(dim=0).argmax()
                argmin = (m.basis ** 2).sum(dim=0).argmin()
                ff = m(self.grid.to(pl_module.device))
                trainer.logger[0].experiment.add_image('online_spectrum_argmax', ff.view(256, 256, -1)[:, :, argmax],
                                                       global_step=trainer.global_step, dataformats='HW')
                trainer.logger[0].experiment.add_image('online_spectrum_argmin', ff.view(256, 256, -1)[:, :, argmin],
                                                       global_step=trainer.global_step, dataformats='HW')

                m = pl_module.generator_ema.fourier_mapping
                argmax = (m.basis ** 2).sum(dim=0).argmax()
                argmin = (m.basis ** 2).sum(dim=0).argmin()
                ff = m(self.grid.to(pl_module.device))
                trainer.logger[0].experiment.add_image('spectrum_argmax', ff.view(256, 256, -1)[:, :, argmax],
                                                       global_step=trainer.global_step, dataformats='HW')
                trainer.logger[0].experiment.add_image('spectrum_argmin', ff.view(256, 256, -1)[:, :, argmin],
                                                       global_step=trainer.global_step, dataformats='HW')


# original https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/callbacks/byol_updates.py
class EMAWeightUpdate(Callback):
    """
    Your model should have:
        - ``self.{{online_model_name}}``
        - ``self.{{target_model_name}}``
    Updates the {{target_model_name}} params using an exponential moving average update rule weighted by tau.
    Example::
        # model must have 2 attributes
        model = Model()
        model.{{online_model_name}} = ...
        model.{{target_model_name}} = ...
        trainer = Trainer(callbacks=[EMAWeightUpdate()])
    """

    def __init__(self, online_model_name: str, target_model_name: str,
                 update_tau: bool = False, initial_tau: float = 0.999):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()
        self.initial_tau = initial_tau
        self.current_tau = initial_tau
        self.online_model_name = online_model_name
        self.target_model_name = target_model_name
        self.update_tau = update_tau

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # get networks
        online_net = getattr(pl_module, self.online_model_name)
        target_net = getattr(pl_module, self.target_model_name)

        # update weights
        self.update_weights(online_net, target_net, self.current_tau)

        # update tau after
        if self.update_tau:
            self.current_tau = self._update_tau(pl_module, trainer)

    def _update_tau(self, pl_module, trainer):
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * pl_module.global_step / max_steps) + 1) / 2
        return tau

    @staticmethod
    def update_weights(online_net, target_net, tau):
        # apply MA weight update
        for (name, online_p), (_, target_p) in zip(online_net.named_parameters(), target_net.named_parameters()):
            target_p.data = tau * target_p.data + (1 - tau) * online_p.data
