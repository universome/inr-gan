from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from firelab.config import Config

from src.models.gan import Discriminator
from src.models.resnet_gan import ResnetDiscriminator
from src.models.stylegan2.discriminator import Discriminator as StyleGAN2Discriminator
from src.models.inrs import SIRENs, FourierINRs, HierarchicalFourierINRs
from src.models.layers import create_activation, SizeSampler, EqualLinear, ScaledLeakyReLU
from src.utils.training_utils import sample_noise
from src.dataloaders.imagenet_vs import fill_by_aspect_ratio


class INRGAN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        DISCR_CLS = {
            'nrgan': Discriminator,
            'gan_stability': ResnetDiscriminator,
            'stylegan2': StyleGAN2Discriminator
        }

        self.generator = INRGenerator(config)
        self.discriminator = DISCR_CLS[config.hp.discriminator.type](config)

    def generator_params_without_inr(self) -> List[nn.Parameter]:
        inr_parameters = set(p for p in self.generator.inr.parameters())
        generator_parameters = set(p for p in self.generator.parameters())

        return [p for p in generator_parameters if not p in inr_parameters]

    def inr_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.generator.inr.parameters()]


class INRGenerator(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.init_inr()
        self.init_model()

    def init_inr(self):
        if self.config.hp.inr.type == 'siren':
            self.inr = SIRENs(self.config)
        elif self.config.hp.inr.type == 'fourier_inr':
            self.inr = FourierINRs(self.config)
        elif self.config.hp.inr.type == 'hier_fourier_inr':
            self.inr = HierarchicalFourierINRs(self.config)
        else:
            raise NotImplementedError(f'Unknown INR type: {self.config.hp.inr.type}')

    def init_model(self):
        input_dim = self.config.hp.generator.z_dim

        if self.config.data.is_conditional:
            input_dim += self.config.hp.generator.class_emb_dim
            self.class_embedder = nn.Embedding(self.config.data.num_classes, self.config.hp.generator.class_emb_dim)
        else:
            self.class_embedder = nn.Identity()

        if self.config.data.is_variable_sized:
            input_dim += self.config.hp.generator.size_sampler.pos_emb_dim
            self.size_sampler = SizeSampler(self.config)
        else:
            self.size_sampler = nn.Identity()

        dims = [input_dim] \
            + [self.config.hp.generator.hid_dim] * self.config.hp.generator.num_layers \
            + [self.inr.num_external_params]

        self.mapping_network = nn.Sequential(
            *[INRGeneratorBlock(dims[i], dims[i+1], self.config.hp.generator.layer, is_first_layer=(i==0)) for i in range(len(dims) - 2)])
        self.connector = nn.Linear(dims[-2], dims[-1])

        # self.connector.weight.data.mul_(np.sqrt(1 / dims[1]))
        if self.config.hp.generator.connector_bias_zero_init:
            self.connector.bias.data.mul_(np.sqrt(1 / dims[1]))

    def forward(self, z: Tensor, img_size: int=None, aspect_ratios: List[float]=None) -> Tensor:
        img_size = self.config.data.target_img_size if img_size is None else img_size
        inrs_weights = self.compute_model_forward(z)

        return self.forward_for_weights(inrs_weights, img_size, aspect_ratios)

    def forward_for_weights(self, inrs_weights: Tensor, img_size: int=None, aspect_ratios: List[float]=None, return_activations: bool=False) -> Tensor:
        img_size = self.config.data.target_img_size if img_size is None else img_size
        generation_result = self.inr.generate_image(
            inrs_weights, img_size, aspect_ratios=aspect_ratios, return_activations=return_activations)

        if return_activations:
            images, inr_activations = generation_result
        else:
            images = generation_result

        if self.config.data.is_variable_sized:
            images = fill_by_aspect_ratio(images, aspect_ratios.cpu().data.tolist(), self.config.hp.generator.resize_strategy, fill_value=-1.0)

        if return_activations:
            return images, inr_activations
        else:
            return images

    def compute_model_forward(self, z: Tensor) -> Tensor:
        latents = self.mapping_network(z)
        weights = self.connector(latents)

        return weights

    def get_output_matrix_size(self) -> int:
        return self.connector.weight.numel()

    def sample_noise(self, batch_size: int, correction: Config=None) -> Tensor:
        return sample_noise(self.config.hp.generator.dist, self.config.hp.generator.z_dim, batch_size, correction)

    def generate_image(self, batch_size: int, device: str, return_activations: bool=False, return_labels: bool=False) -> Tensor:
        """
        Generates an INR and computes it
        """
        inputs = self.sample_noise(batch_size).to(device) # [batch_size, z_dim]
        labels = None # In case of conditional generation
        aspect_ratios = None # In case of variable-sized generation

        if self.config.data.is_conditional:
            labels = torch.randint(0, self.config.data.num_classes, size=(batch_size,)).to(device) # [batch_size]
            class_embs = self.class_embedder(labels) # [batch_size, class_emb_dim]
            inputs = torch.cat([inputs, class_embs], dim=1) # [batch_size, z_dim + class_emb_dim]

        if self.config.data.is_variable_sized:
            aspect_ratios = self.size_sampler.sample_aspect_ratios(labels) # [batch_size]
            aspect_ratios_embs = self.size_sampler.pos_embedder(aspect_ratios.unsqueeze(1)) # [batch_size, aspect_ratio_emb_dim]
            inputs = torch.cat([inputs, aspect_ratios_embs], dim=1) # [batch_size, z_dim + class_emb_dim + aspect_ratio_emb_dim]

        if return_activations:
            gen_activations = {}
            x = inputs

            for i, module in enumerate(self.mapping_network):
                x = module(x)
                gen_activations[f'MappingNetwork-Layer-{i}'] = x.cpu().detach()

            x = self.connector(x)
            gen_activations[f'Connector'] = x.cpu().detach()

            inr_params = x
        else:
            inr_params = self.compute_model_forward(inputs)

        # Generating the images
        generation_result = self.forward_for_weights(
            inr_params, aspect_ratios=aspect_ratios, return_activations=return_activations)

        if return_activations:
            images, inr_activations = generation_result

            if return_labels:
                return images, labels, gen_activations, inr_activations
            else:
                return images, gen_activations, inr_activations
        else:
            images = generation_result

            if return_labels:
                return images, labels
            else:
                return images


class INRGeneratorBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, config: Config, is_first_layer: bool):
        super().__init__()

        if config.equalized_lr:
            layers = [EqualLinear(in_features, out_features)]
        else:
            layers = [nn.Linear(in_features, out_features)]

        if in_features == out_features and config.residual and not is_first_layer:
            self.residual = True
            self.main_branch_weight = nn.Parameter(torch.tensor(config.main_branch_weight))
        else:
            self.residual = False

        if config.has_bn:
            layers.append(nn.BatchNorm1d(out_features))

        layers.append(create_activation(config.activation, **config.activation_kwargs))

        self.transform = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        y = self.transform(x)

        if self.residual:
            return x + self.main_branch_weight * y
        else:
            return y
