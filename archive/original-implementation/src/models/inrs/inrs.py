from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.models.layers import ScaledLinear, create_activation, Sine
from src.models.inrs.modules import (
    INRModule,
    INRProxy,
    INRLinear,
    INRSELinear,
    INRSharedLinear,
    INRmFiLM,
    INRActNorm,
    INRAdaIN,
    INRFactorizedLinear,
    INRFactorizedSELinear,
    INRResidual,
    INRSequential,
    INRInputSkip,
    MultiModalINRFactorizedSELinear,
    MultiModalINRSharedLinear,
    INRSVDLinear,
    INRResConnector,
    INRModuleDict,
    INRIdentity,
    INRPixelNorm,
    INRToRGB,
    INRNoiseInjection,
    INRFourierFeats,
    INRCoordsSkip,
)
from src.models.init import compute_siren_std, compute_old_siren_std, compute_dense_scale
from src.utils.training_utils import generate_coords, generate_random_resolution_coords


class INRs(nn.Module):
    def __init__(self, config: Config):
        super(INRs, self).__init__()

        self.config = config
        self.init_model()
        self.num_external_params = sum(m.num_external_params for m in self.model.children())
        self.num_shared_params = sum(p.numel() for p in self.parameters())
        self.min_scale = 1.0 # In some setting, it is going to be changed over time

    def init_model(self):
        raise NotImplementedError("It is a base class. Implement in `.init_model()` in your child class.")

    @torch.no_grad()
    def generate_input_coords(self, batch_size: int, img_size: int) -> Tensor:
        """
        @param batch_size
        @return coords # [batch_size, coord_dim, n_coords]
        """
        if self.training and self.config.hp.get('progressive_transform.enabled'):
            coords = generate_random_resolution_coords(batch_size, img_size, min_scale=self.min_scale)
        elif self.training and self.config.data.get('concat_patches.enabled'):
            normal_coords = generate_coords(batch_size, img_size)
            patch_coords = generate_random_resolution_coords(batch_size, img_size, scale=self.config.data.concat_patches.ratio)
            coords = torch.cat([normal_coords, patch_coords], dim=2)
        else:
            coords = generate_coords(batch_size, img_size)

        if self.config.hp.inr.input_coord_range_lhs == 0:
            pass
        elif self.config.hp.inr.input_coord_range_lhs == -1:
            coords = coords * 2 - 1
        else:
            raise NotImplementedError(f'Unknown `input_coord_range_lhs`: {self.config.hp.inr.input_coord_range_lhs}')

        return coords

    def generate_image(self, inrs_weights: Tensor, img_size: int, return_activations: bool=False) -> Tensor:
        coords = self.generate_input_coords(len(inrs_weights), img_size).to(inrs_weights.device)

        if return_activations:
            images_raw, activations = self.forward(coords, inrs_weights, return_activations=True) # [batch_size, num_channels, num_coords]
        else:
            images_raw = self.forward(coords, inrs_weights) # [batch_size, num_channels, num_coords]

        if self.training and self.config.data.get('concat_patches.enabled'):
            images_raw, patches_raw = images_raw[:, :, :img_size ** 2], images_raw[:, :, img_size ** 2:]
            images = images_raw.view(len(inrs_weights), self.config.data.num_img_channels, img_size, img_size) # [batch_size, num_channels, img_size, img_size]
            patches = patches_raw.view(len(inrs_weights), self.config.data.num_img_channels, img_size, img_size) # [batch_size, num_channels, img_size, img_size]

            images = torch.cat([images, patches], dim=1) # [batch_size, num_channels * 2, img_size, img_size]
        else:
            images = images_raw.view(len(inrs_weights), self.config.data.num_img_channels, img_size, img_size) # [batch_size, num_channels, img_size, img_size]

        return (images, activations) if return_activations else images

    def apply_weights(self, x: Tensor, inrs_weights: Tensor, return_activations: bool=False) -> Tensor:
        curr_w = inrs_weights

        if return_activations:
            activations = {'initial': x.cpu().detach()}

        for i, module in enumerate(self.model.children()):
            module_params = curr_w[:, :module.num_external_params]
            x = module(x, module_params)
            curr_w = curr_w[:, module.num_external_params:]

            if return_activations:
                activations[f'{i}-{module}'] = x.cpu().detach()

        assert curr_w.numel() == 0, f"Not all params were used: {curr_w.shape}"

        return (x, activations) if return_activations else x

    def forward(self, coords: Tensor, inrs_weights: Tensor, return_activations: bool=False) -> Tensor:
        """
        Computes a batch of INRs in the given coordinates

        @param coords: coordinates | [n_coords, 2]
        @param inrs_weights: weights of INRs | [batch_size, coord_dim]
        """
        return self.apply_weights(coords, inrs_weights, return_activations=return_activations)

    def create_transform(self, in_features: int, out_features: int, layer_type: str='linear',
                         is_coord_layer: bool=False, weight_std: float=None, bias_std: float=None):
        TYPE_TO_INR_CLASS = {
            'linear': INRLinear,
            'se_linear': INRSELinear,
            'shared_linear': INRSharedLinear,
            'adain_linear': INRSharedLinear,
            'mfilm': INRmFiLM,
            'factorized': INRFactorizedLinear,
            'se_factorized': INRFactorizedSELinear,
            'mm_se_factorized': MultiModalINRFactorizedSELinear,
            'mm_shared_linear': MultiModalINRSharedLinear,
            'svd_linear': INRSVDLinear,
        }

        weight_std = self.compute_weight_std(in_features, is_coord_layer) if weight_std is None else weight_std
        bias_std = self.compute_bias_std(in_features, is_coord_layer) if bias_std is None else bias_std
        layers = [TYPE_TO_INR_CLASS[layer_type](
            in_features,
            out_features,
            weight_std=weight_std,
            bias_std=bias_std,
            **self.config.hp.inr.get(f'module_kwargs.{layer_type}', {}))]

        if layer_type == 'adain_linear':
            layers.append(INRAdaIN(out_features))

        return layers

    def compute_weight_std(self, in_features: int, is_coord_layer: bool) -> float:
        raise NotImplementedError

    def compute_bias_std(self, in_features: int, is_coord_layer: bool) -> float:
        if is_coord_layer:
            additional_scale = self.config.hp.inr.bias_coords_layer_std_scale
        else:
            additional_scale = self.config.hp.inr.bias_hid_layer_std_scale

        return self.compute_weight_std(in_features, is_coord_layer) * additional_scale


class SIRENs(INRs):
    def init_model(self):
        layers = self.create_transform(
            self.config.hp.inr.coord_dim,
            self.config.hp.inr.layer_sizes[0],
            layer_type=self.config.hp.inr.coords_layer_type, # First layer is of full control
            is_coord_layer=True)
        layers.append(INRProxy(self.create_sine(self.config.hp.inr.w0_initial)))

        for i in range(len(self.config.hp.inr.layer_sizes) - 1):
            layers.extend(self.create_transform(
                self.config.hp.inr.layer_sizes[i],
                self.config.hp.inr.layer_sizes[i+1],
                layer_type=self.config.hp.inr.hid_layer_type)) # Middle layers are large so they are controlled via AdaIN
            layers.append(INRProxy(self.create_sine(self.config.hp.inr.w0)))

        layers.extend(self.create_transform(
            self.config.hp.inr.layer_sizes[-1],
            self.config.data.num_img_channels,
            layer_type='linear' # The last layer is small so let's also control it fully
        ))
        layers.append(INRProxy(create_activation(self.config.hp.inr.output_activation)))

        self.model = nn.Sequential(*layers)

    def create_sine(self, w0: float) -> Sine:
        if self.config.hp.inr.init_type == 'old_with_sine_scale':
            return Sine(scale=w0)
        else:
            return Sine(scale=1.0)

    def compute_weight_std(self, in_features: int, is_coord_layer: bool) -> float:
        if self.config.hp.inr.init_type == 'old_with_sine_scale':
            w0 = self.config.hp.inr.w0_initial if is_coord_layer else self.config.hp.inr.w0
            weight_std = compute_old_siren_std(in_features, w0)
        elif self.config.hp.inr.init_type == 'old':
            weight_std = np.sqrt(2 / in_features)
        else:
            weight_std = compute_siren_std(in_features, is_coord_layer)


class FourierINRs(INRs):
    def init_model(self):
        layers = self.create_transform(
            self.config.hp.inr.coord_dim,
            self.config.hp.inr.layer_sizes[0] // 2,
            layer_type=self.config.hp.inr.coords_layer_type,
            is_coord_layer=True)
        layers.append(INRProxy(create_activation('sines_cosines')))

        hid_layers = []

        for i in range(len(self.config.hp.inr.layer_sizes) - 1):
            if self.config.hp.inr.get('skip_coords') and i > 0:
                input_dim = self.config.hp.inr.layer_sizes[i] + self.config.hp.inr.layer_sizes[0]
            else:
                input_dim = self.config.hp.inr.layer_sizes[i]

            curr_transform_layers = self.create_transform(
                input_dim,
                self.config.hp.inr.layer_sizes[i+1],
                layer_type=self.config.hp.inr.hid_layer_type)
            curr_transform_layers.append(INRProxy(create_activation(self.config.hp.inr.activation)))

            if self.config.hp.inr.residual:
                hid_layers.append(INRResidual(INRSequential(*curr_transform_layers)))
            else:
                hid_layers.extend(curr_transform_layers)

        if self.config.hp.inr.get('skip_coords'):
            layers.append(INRInputSkip(*hid_layers))
        else:
            layers.extend(hid_layers)

        layers.extend(self.create_transform(self.config.hp.inr.layer_sizes[-1], self.config.data.num_img_channels, 'linear'))
        layers.append(INRProxy(create_activation(self.config.hp.inr.output_activation)))

        self.model = nn.Sequential(*layers)

    def compute_weight_std(self, in_features: int, is_coord_layer: bool) -> float:
        if is_coord_layer:
            return self.config.hp.inr.fourier_scale
        else:
            return np.sqrt(2 / in_features)


class StaticINRs(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        if self.config.hp.inr.type == "siren":
            self.inrs = SIRENs(self.config)
        elif self.config.hp.inr.type == "fourier_inr":
            self.inrs = FourierINRs(self.config)
        elif self.config.hp.inr.type == "hier_fourier_inr":
            self.inrs = HierarchicalFourierINRs(self.config)
        else:
            raise NotImplementedError(f'Unknown INR type: {self.config.hp.inr.type}')

        self.external_params = nn.Parameter(torch.randn(
            self.config.hp.num_inrs, self.inrs.num_external_params))

    def forward(self, coords: Tensor, return_activations: bool=False) -> Tensor:
        return self.inrs(coords, self.external_params, return_activations=return_activations)


class HierarchicalFourierINRs(FourierINRs, INRs):
    """
    Hierarchical INRs operate in a bit different regime:
    we first generate 8x8 resolution, then 16x16, etc...
    This allows us to use larger layer sizes at the beginning
    """
    def __init__(self, config: Config):
        nn.Module.__init__(self)

        self.config = config
        self.init_model()
        self.num_external_params = self.model.num_external_params
        self.num_shared_params = sum(p.numel() for p in self.parameters())

        print('Resolutions:', self.generate_img_sizes(self.config.data.target_img_size))

        assert not self.config.data.get('concat_patches.enabled')

    def init_model(self):
        blocks = []
        if self.config.hp.inr.res_increase_scheme.enabled:
            res_configs = [self.create_res_config(i) for i in range(self.config.hp.inr.num_blocks)]
        else:
            resolutions = self.generate_img_sizes(self.config.data.target_img_size)
            res_configs = [self.config.hp.inr.resolutions_params[resolutions[i]] for i in range(self.config.hp.inr.num_blocks)]

        print('resolution:', [c.resolution for c in res_configs])
        print('dim:', [c.dim for c in res_configs])
        print('num_learnable_coord_feats:', [c.num_learnable_coord_feats for c in res_configs])
        print('to_rgb:', [c.to_rgb for c in res_configs])
        num_to_rgb_blocks = sum(c.to_rgb for c in res_configs)

        for i, res_config in enumerate(res_configs):
            # 1. Creating coord fourier feat embedders for each resolution
            coord_embedder = INRSequential(
                INRFourierFeats(res_config),
                INRProxy(create_activation('sines_cosines')))
            coord_feat_dim = coord_embedder[0].get_num_feats() * 2

            # 2. Main branch. First need does not need any wiring, but later layers use it.
            # A good thing is that we do not need skip-coords anymore.
            if i > 0:
                # Different-resolution blocks are wired together with the connector
                connector_layers = [INRResConnector(
                    res_configs[i-1].dim,
                    coord_feat_dim,
                    res_config.dim,
                    self.config.hp.inr.upsampling_mode,
                    **self.config.hp.inr.module_kwargs.se_factorized),
                ]
                if self.config.hp.inr.use_pixel_norm:
                    connector_layers.append(INRPixelNorm())
                connector_layers.append(INRProxy(create_activation(self.config.hp.inr.activation, **self.config.hp.inr.activation_kwargs)))
                connector = INRSequential(*connector_layers)
            else:
                connector = INRIdentity()

            transform_layers = []
            for j in range(res_config.n_layers):
                if i == 0 and j == 0:
                    input_size = coord_feat_dim # Since we do not have previous feat dims
                elif self.config.hp.inr.skip_coords:
                    input_size = coord_feat_dim + res_config.dim
                else:
                    input_size = res_config.dim

                transform_layers.extend(self.create_transform(
                    input_size, res_config.dim,
                    layer_type=self.config.hp.inr.hid_layer_type))

                if self.config.hp.inr.use_pixel_norm:
                    transform_layers.append(INRPixelNorm())

                if self.config.hp.inr.use_noise:
                    transform_layers.append(INRNoiseInjection())

                transform_layers.append(INRProxy(create_activation(self.config.hp.inr.activation, **self.config.hp.inr.activation_kwargs)))

            if res_config.to_rgb or i == (self.config.hp.inr.num_blocks - 1):
                to_rgb_weight_std = self.compute_weight_std(res_config.dim, is_coord_layer=False)
                to_rgb_bias_std = self.compute_bias_std(res_config.dim, is_coord_layer=False)

                if self.config.hp.inr.additionaly_scale_to_rgb:
                    to_rgb_weight_std /= np.sqrt(num_to_rgb_blocks)
                    to_rgb_bias_std /= np.sqrt(num_to_rgb_blocks)

                to_rgb = INRToRGB(
                    res_config.dim,
                    self.config.hp.inr.to_rgb_activation,
                    self.config.hp.inr.upsampling_mode,
                    to_rgb_weight_std,
                    to_rgb_bias_std)
            else:
                to_rgb = INRIdentity()

            if self.config.hp.inr.skip_coords:
                transform = INRCoordsSkip(*transform_layers, concat_to_the_first=i > 0)
            else:
                transform = INRSequential(*transform_layers)

            blocks.append(INRModuleDict({
                'coord_embedder': coord_embedder,
                'transform': transform,
                'connector': connector,
                'to_rgb': to_rgb,
            }))

        self.model = INRModuleDict({f'b_{i}': b for i, b in enumerate(blocks)})

    def create_res_config(self, block_idx: int) -> Config:
        increase_conf = self.config.hp.inr.res_increase_scheme
        num_blocks = self.config.hp.inr.num_blocks
        resolutions = self.generate_img_sizes(self.config.data.target_img_size)
        fourier_scale = np.linspace(increase_conf.fourier_scales.min, increase_conf.fourier_scales.max, num_blocks)[block_idx]
        dim = np.linspace(increase_conf.dims.max, increase_conf.dims.min, num_blocks).astype(int)[block_idx]
        num_coord_feats = np.linspace(increase_conf.num_coord_feats.max, increase_conf.num_coord_feats.min, num_blocks).astype(int)[block_idx]

        return Config({
            'resolution': resolutions[block_idx],
            'num_learnable_coord_feats': num_coord_feats.item(),
            'use_diag_feats': resolutions[block_idx] <= increase_conf.diag_feats_threshold,
            'max_num_fixed_coord_feats': 10000 if increase_conf.use_fixed_coord_feats else 0,
            'dim': dim.item(),
            'fourier_scale': fourier_scale.item(),
            'to_rgb': resolutions[block_idx] >= increase_conf.to_rgb_res_threshold,
            'n_layers': 1
        })

    def generate_image(self, inrs_weights: Tensor, img_size: int, aspect_ratios=None, return_activations: bool=False) -> Tensor:
        # Generating coords for each resolution
        batch_size = len(inrs_weights)
        img_sizes = self.generate_img_sizes(img_size)

        coords_list = [generate_coords(batch_size, s) for s in img_sizes] # (num_blocks, [batch_size, 2, num_coords_in_block])

        if return_activations:
            images_raw, activations = self.forward(coords_list, inrs_weights, return_activations=True) # [batch_size, num_channels, num_coords]
        else:
            images_raw = self.forward(coords_list, inrs_weights) # [batch_size, num_channels, num_coords]

        images = images_raw.view(batch_size, self.config.data.num_img_channels, img_size, img_size) # [batch_size, num_channels, img_size, img_size]

        return (images, activations) if return_activations else images

    def apply_weights(self,
                      coords_list: List[Tensor],
                      inrs_weights: Tensor,
                      return_activations: bool=False,
                      noise_injections: List[Tensor]=None) -> Tensor:

        device = inrs_weights.device
        curr_w = inrs_weights
        images = None

        if return_activations:
            activations = {}

        for i in range(self.config.hp.inr.num_blocks):
            coords = coords_list[i].to(device)
            block = self.model[f'b_{i}']
            curr_w, coord_feats = self.apply_module(curr_w, block.coord_embedder, coords)

            if i == 0:
                if self.config.hp.inr.skip_coords:
                    curr_w, x = self.apply_module(curr_w, block.transform, coord_feats, coord_feats)
                else:
                    curr_w, x = self.apply_module(curr_w, block.transform, coord_feats)
            else:
                # Apply a connector
                curr_w, x = self.apply_module(curr_w, block.connector[0], x, coord_feats) # transform
                if return_activations:
                    activations[f'block-{i}-connector'] = x.cpu().detach()
                curr_w, x = self.apply_module(curr_w, block.connector[1], x) # activation

                # Apply a transform
                if self.config.hp.inr.skip_coords:
                    curr_w, x = self.apply_module(curr_w, block.transform, x, coord_feats)
                else:
                    curr_w, x = self.apply_module(curr_w, block.transform, x)

            if return_activations:
                activations[f'block-{i}'] = x.cpu().detach()

            if isinstance(block.to_rgb, INRToRGB):
                # Converting to an image (possibly using the skip)
                curr_w, images = self.apply_module(curr_w, block.to_rgb, x, images)

                if return_activations:
                    activations[f'block-{i}-images'] = images.cpu().detach()

        if self.config.hp.inr.output_activation == 'tanh':
            out = images.tanh()
        elif self.config.hp.inr.output_activation == 'clamp':
            out = images.clamp(-1.0, 1.0)
        elif self.config.hp.inr.output_activation == 'sigmoid':
            out = images.sigmoid() * 2 - 1
        elif self.config.hp.inr.output_activation in ['none', None]:
            out = images
        else:
            raise NotImplementedError(f'Unknown output activation: {self.config.hp.inr.output_activation}')

        if return_activations:
            activations[f'block-final'] = out.cpu().detach()

        assert curr_w.numel() == 0, f"Not all params were used: {curr_w.shape}"

        return (out, activations) if return_activations else out

    def apply_module(self, curr_w: Tensor, module: INRModule, *inputs) -> Tuple[Tensor, Tensor]:
        """
        Applies params and returns the remaining ones
        """
        module_params = curr_w[:, :module.num_external_params]
        y = module(*inputs, module_params)
        remaining_w = curr_w[:, module.num_external_params:]

        return remaining_w, y

    def generate_img_sizes(self, target_img_size: int) -> List[int]:
        """
        Generates coord features for each resolution to produce a final image
        The main logic is in computing resolutions
        """
        if self.config.hp.inr.res_increase_scheme.enabled:
            return self.generate_linear_img_sizes(target_img_size)
        else:
            return self.generate_exp_img_sizes(target_img_size)

    def generate_exp_img_sizes(self, target_img_size: int) -> List[int]:
        # This determines an additional upscale factor for the upscaling block
        extra_upscale_factor = target_img_size // self.config.data.target_img_size
        img_sizes = []
        curr_img_size = target_img_size

        for i in range(self.config.hp.inr.num_blocks - 1, -1, -1):
            img_sizes.append(curr_img_size)

            if i == self.config.hp.inr.upsample_block_idx:
                curr_img_size = curr_img_size // (extra_upscale_factor * 2)
            else:
                curr_img_size = curr_img_size // 2

        return list(reversed(img_sizes))

    def generate_linear_img_sizes(self, target_img_size: int) -> List[int]:
        min_size = self.config.hp.inr.res_increase_scheme.min_resolution
        max_size = target_img_size
        img_sizes = np.linspace(min_size, max_size, self.config.hp.inr.num_blocks).astype(int)

        return img_sizes.tolist()

    def forward(self, coords_list: List[Tensor], inrs_weights: Tensor, return_activations: bool=False) -> Tensor:
        """
        Computes a batch of INRs in the given coordinates

        @param coords_list: coordinates | (num_blocks, [batch_size, n_coords, 2])
        @param inrs_weights: weights of INRs | [batch_size, coord_dim]
        """
        return self.apply_weights(coords_list, inrs_weights, return_activations=return_activations)
