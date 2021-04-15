from typing import Callable, Optional, Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from firelab.config import Config

from src.models.layers import blur_upsample


class INRModule(nn.Module):
    """
    This is a base class for modules that are to be used as a part of an INR
    """
    def __init__(self):
        super(INRModule, self).__init__()

        # Number of params that should be provided by a hypernetwork
        self.num_external_params = 0

    def _check_params_size(self, params: Tensor) -> bool:
        """
        Checks that we have provided a proper amount of parameters
        @param params: INRs weights # [num_inrs, num_external_params]
        """
        assert params.shape[1] == self.num_external_params, \
            f"Wrong shape: {params.shape}. Num external params: {self.num_external_params}"

        return True

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        """
        params:
        - x: input data of size [num_inrs, hid_dim, num_coords]
        - params: external params of size [num_inrs, num_external_params]
        """
        raise NotImplementedError("You should implement .forward() method!")

    def __repr__(self) -> str:
        raise NotImplementedError(
            'Sorry, but you should manually implement .__repr__() method. '\
            'Otherwise, we will have unreadable activations histograms.')


class INRModuleDict(INRModule):
    def __init__(self, items: Dict[str, INRModule]):
        super(INRModuleDict, self).__init__()

        self._inr_modules = nn.ModuleDict(items)
        self.num_external_params = sum(m.num_external_params for m in self._inr_modules.values())

        for k, v in self._inr_modules.items():
            setattr(self, k, v)

    def __getitem__(self, k: str) -> INRModule:
        return self._inr_modules[k]

    def __setitem__(self, k: str, v: INRModule):
        self._inr_modules[k] = v
        setattr(self, k, v)

    def __repr__(self) -> str:
        return f'INRModuleDict: {"-".join([k for k in self._inr_modules])}'


class INRIdentity(INRModule):
    def forward(self, x, params):
        return x

    def __repr__(self) -> str:
        return f'INRIdentity'


class INRLinear(INRModule):
    def __init__(self, in_features: int, out_features: int, weight_std: Optional[float]=None,
                       bias_std: Optional[float]=None):
        super(INRLinear, self).__init__()

        self.weight_size = out_features * in_features
        self.weight_shape = (out_features, in_features)
        self.bias_size = out_features
        self.weight_std = self.compute_weight_std(in_features) if weight_std is None else weight_std
        self.bias_std = self.compute_bias_std(in_features) if bias_std is None else bias_std
        self.num_external_params = self.weight_size + self.bias_size

    def compute_weight_std(self, in_features: int) -> float:
        """
        Computes std for kaiming init, like in default torch Linear/Conv2d
        """
        assert False, "You should provide your own weight std."
        gain = nn.init.calculate_gain('leaky_relu', np.sqrt(5))
        std = gain / np.sqrt(in_features)

        return std

    def compute_bias_std(self, in_features: int) -> float:
        assert False, "You should provide your own bias std."
        return 1 / np.sqrt(in_features)

    def transform_params(self, params: Tensor) -> Tuple[Tensor, Tensor]:
        num_inrs = params.shape[0]
        weights = self.weight_std * params[:, :self.weight_size] # [num_inrs, W_size]
        weights = weights.view(num_inrs, *self.weight_shape) # [num_inrs, out_features, in_features]

        biases = self.bias_std * params[:, -self.bias_size:] # [num_inrs, out_features]
        biases = biases.view(num_inrs, self.bias_size, 1) # [num_inrs, out_features, 1]

        return weights, biases

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        self._check_params_size(params)

        weights, biases = self.transform_params(params)
        out = torch.bmm(weights, x) + biases # [num_inrs, out_features, n_coords]

        return out

    def __repr__(self) -> str:
        return f'INRLinear {self.weight_shape[1]} -> {self.weight_shape[0]}'


class INRSELinear(INRLinear):
    """
    This module combines the idea of INRLinear, but it also
    applies weights from the generator in a squeeze-and-excitation like fashion:
    It computes the final weight value via: params_own \odot \sigma(params_gen),
    where \sigma is either sigmoid(x) * 2 or tanh(x) + 1 (so it is centered around 1)
    """
    def __init__(self, in_features: int, out_features: int, weight_std: float, bias_std: float, init_dist: str='normal'):
        super(INRSELinear, self).__init__(in_features, out_features, weight_std=weight_std, bias_std=bias_std)

        self.weight_std = weight_std
        self.bias_std = bias_std
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        initialize_weights_(self.weight, 1.0, init_dist)
        initialize_weights_(self.bias, 1.0, init_dist)

    def transform_params(self, params: Tensor) -> Tensor:
        weights_gen, biases_gen = INRLinear.transform_params(self, params)
        weights_gen, biases_gen = weights_gen.sigmoid() * 2, biases_gen.sigmoid() * 2

        W_shared = self.weight_std * self.weight.view(1, *self.weight.shape) # [num_inrs, out_features, in_features]
        b_shared = self.bias_std * self.bias.view(1, self.bias_size, 1) # [num_inrs, out_features, 1]

        # Applying "squeeze-and-excitation"
        weights = weights_gen * W_shared # [num_inrs, out_features, in_features]
        biases = biases_gen * b_shared # [num_inrs, out_features, 1]

        return weights, biases

    def __repr__(self) -> str:
        return f'INRSELinear {self.weight.shape[1]} -> {self.weight.shape[0]}'


class INRSharedLinear(INRModule):
    def __init__(self, in_features: int, out_features: int, weight_std: float, bias_std: float, init_dist: str='normal'):
        super(INRSharedLinear, self).__init__()

        self.weight_std = weight_std
        self.bias_std = bias_std
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        initialize_weights_(self.weight, 1.0, init_dist)
        initialize_weights_(self.bias, 1.0, init_dist)

        self.num_external_params = 0

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        self._check_params_size(params)

        inputs = x.permute(0, 2, 1) # [num_inrs, num_coords, in_features]
        out = F.linear(inputs, self.weight * self.weight_std, bias=self.bias * self.bias_std) # [num_inrs, num_coords, out_features]
        out = out.permute(0, 2, 1) # [num_inrs, out_features, num_coords]

        return out

    def __repr__(self) -> str:
        return f'INRSharedLinear {self.model.weight.shape[1]} -> {self.model.weight.shape[0]}'


class INRmFiLM(INRLinear):
    """
    mFiLM transfrom type from https://arxiv.org/abs/2006.07543
    """
    def __init__(self, in_features: int, out_features: int, weight_std: float, bias_std: float, init_dist: str="normal"):
        INRModule.__init__(self)

        self.out_features = out_features
        self.weight_std = weight_std
        self.bias_std = bias_std
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        initialize_weights_(self.weight, 1.0, init_dist)
        initialize_weights_(self.bias, 1.0, init_dist)

        self.num_external_params = 3 * out_features # n_scales + n_shifts + n_biases

    def transform_params(self, params: Tensor) -> Tensor:
        """
        @param params: [num_inrs, 3 * out_features]
        """
        num_inrs = len(params)
        scales = params[:, :self.out_features].view(num_inrs, self.out_features, 1) / 10 + 1
        shifts = params[:, self.out_features : -self.out_features].view(num_inrs, self.out_features, 1) * self.weight_std / 10
        biases = params[:, -self.out_features:].view(num_inrs, self.out_features, 1) * self.bias_std / 10

        W_shared = self.weight_std * self.weight
        b_shared = self.bias_std * self.bias
        weights_final = W_shared * scales + shifts
        biases_final = b_shared.view(1, self.out_features, 1) + biases

        return weights_final, biases_final

    def __repr__(self) -> str:
        return f'INRmFiLM {self.weight.shape[1]} -> {self.weight.shape[0]}'


class INRFactorizedLinear(INRLinear):
    """
    Keeps a matrix in a factorized form
    """
    def __init__(self, in_features: int, out_features: int, weight_std: float, bias_std: float, rank: int=3):
        INRModule.__init__(self)

        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.weight_std = weight_std
        self.bias_std = bias_std

        self.num_external_params = rank * in_features + rank * out_features + out_features # lhs + rhs + bias

    def transform_params(self, params: Tensor) -> Tensor:
        num_inrs = len(params)
        lhs = params[:, :self.out_features * self.rank]
        rhs = params[:, self.in_features * self.rank:-self.out_features]
        bias = params[:, -self.out_features:]

        lhs = lhs.view(num_inrs, self.out_features, self.rank)
        rhs = rhs.view(num_inrs, self.rank, self.in_features)
        weight = torch.bmm(lhs, rhs) * self.weight_std # [num_inrs, out_features, in_features]
        bias = bias.view(num_inrs, self.out_features, 1) * self.bias_std

        return weight, bias

    def __repr__(self) -> str:
        return f'INRFactorizedLinear {self.weight.shape[1]} -> {self.weight.shape[0]}'


class INRFactorizedSELinear(INRLinear):
    """
    Keeps a matrix in a factorized form and applies it via squeeze-and-excitation
    """
    def __init__(self, in_features: int, out_features: int, weight_std: float, bias_std: float, rank: int, init_dist: str="normal", equalized_lr: bool=True):
        INRModule.__init__(self)

        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.weight_std = weight_std
        self.bias_std = bias_std
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.equalized_lr = equalized_lr

        if self.equalized_lr:
            magic_constant = 1.0 # N(0, 1/1000000) => scale by (w_scale * 1000000).
            initialize_weights_(self.weight, 1 / np.sqrt(magic_constant), init_dist)
            self.weight_std = weight_std * np.sqrt(magic_constant)
        else:
            initialize_weights_(self.weight, weight_std, init_dist)

        self.num_external_params = rank * in_features + rank * out_features + out_features # lhs + rhs + bias

    def process_external_params(self, params: Tensor) -> Tensor:
        num_inrs = len(params)
        lhs = params[:, :self.out_features * self.rank] # Left matrix
        rhs = params[:, self.out_features * self.rank:-self.out_features] # Right matrix
        bias = params[:, -self.out_features:]

        lhs = lhs.view(num_inrs, self.out_features, self.rank)
        rhs = rhs.view(num_inrs, self.rank, self.in_features)
        weight_modulation = torch.bmm(lhs, rhs) # [num_inrs, out_features, in_features]
        weight_modulation = weight_modulation.sigmoid() * 2

        bias = bias.view(num_inrs, self.out_features, 1) * self.bias_std

        return weight_modulation, bias

    def transform_params(self, params: Tensor) -> Tensor:
        weight_modulation, bias = self.process_external_params(params)
        if self.equalized_lr:
            W_shared = self.weight_std * self.weight
        else:
            W_shared = self.weight
        weight = W_shared.unsqueeze(0) * weight_modulation

        return weight, bias

    def __repr__(self) -> str:
        return f'INRFactorizedSELinear {self.weight.shape[1]} -> {self.weight.shape[0]}'


class INRAdaIN(INRModule):
    def __init__(self, size: int) -> Tensor:
        super(INRAdaIN, self).__init__()

        self.size = size
        self.num_external_params = self.size * 2

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        """
        @param x: [num_inrs, hid_dim, num_coords]
        @param params: [num_inrs, num_external_params]
        """
        self._check_params_size(params)
        num_inrs, hid_dim = x.shape[0], x.shape[1]
        scales = params[:, :self.size].view(num_inrs, hid_dim, 1) # [num_inrs, hid_dim, 1]
        biases = params[:, self.size:].view(num_inrs, hid_dim, 1) # [num_inrs, hid_dim, 1]

        out = x * scales.sigmoid() * 2 + biases

        return out

    def __repr__(self) -> str:
        return f'INRAdaIN {self.size}'


class INRProxy(INRModule):
    """
    A proxy layer which just applies another function in INRModule-like fashion
    """
    def __init__(self, fn: Callable):
        super(INRProxy, self).__init__()

        self.fn = fn

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        return self.fn(x)

    def __repr__(self) -> str:
        return f"INR {type(self.fn).__name__}"


class INRActNorm(INRModule):
    """
    ActNorm from Glow paper.
    Adapted from https://github.com/rosinality/glow-pytorch/blob/master/model.py
    """
    def __init__(self, size: int):
        super(INRActNorm, self).__init__()

        self.bias = nn.Parameter(torch.zeros(1, size, 1))
        self.scale = nn.Parameter(torch.ones(1, size, 1))

        self.register_buffer('initialized', torch.tensor(False))

    def initialize(self, x: Tensor) -> Tensor:
        """
        Selects parameters based on the given batch (assuming it is big enough)
        @param x if size [num_inrs, size, n_coords]
        """
        with torch.no_grad():
            self.bias.data.copy_(-x.mean(dim=(0,2), keepdim=True))
            self.scale.data.copy_(1 / (x.std(dim=(0,2), keepdim=True) + 1e-6))

        self.initialized.fill_(True)

    def forward(self, x, params: Tensor) -> Tensor:
        assert params.numel() == 0, f"Wrong shape: {params.shape}"

        if not self.initialized:
            self.initialize(x)

        return self.scale * x + self.bias

    def __repr__(self) -> str:
        return f'INRActNorm {self.bias.shape}'


class INRResidual(INRModule):
    def __init__(self, transform: INRModule, res_branch_weight: float=0.01):
        super(INRResidual, self).__init__()

        self.transform = transform
        self.num_external_params = 1 + self.transform.num_external_params
        self.res_branch_weight = res_branch_weight

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        # res_branch_weight = (params[:, 0] - 2).sigmoid() # So it is near 0.1 (i.e. switched off) initially
        res_branch_weight = (params[:, 0] * 0.01 + self.res_branch_weight).view(len(x), 1, 1)
        res_branch_weight = res_branch_weight.view(len(x), 1, 1)
        main_out = self.transform(x, params[:, 1:])

        return res_branch_weight * x + main_out

    def __repr__(self) -> str:
        return f'INRResidual [{self.transform}]'


class INRSequential(INRModule):
    def __init__(self, *modules):
        super(INRSequential, self).__init__()

        self._inr_modules = nn.ModuleList(modules)
        self.num_external_params = sum(c.num_external_params for c in self._inr_modules)

    def __len__(self) -> int:
        return len(self._inr_modules)

    def __getitem__(self, i: int) -> INRModule:
        return self._inr_modules[i]

    def forward(self, x, params: Tensor) -> Tensor:
        curr_w = params

        for module in self._inr_modules:
            module_params = curr_w[:, :module.num_external_params]
            x = module(x, module_params)
            curr_w = curr_w[:, module.num_external_params:]

        return x

    def __repr__(self) -> str:
        return f'INRSequential[{len(self)}]'


class INRInputSkip(INRModule):
    """
    INR architecture which adds a skip-connection
    of input coordinates to each subsequent layer
    """
    def __init__(self, *modules, skip_type: str='residual'):
        super(INRInputSkip, self).__init__()

        self.skip_type = skip_type
        self._inr_modules = nn.ModuleList(modules)

        self.num_external_params = sum(c.num_external_params for c in self._inr_modules)

    def __len__(self) -> int:
        return len(self._inr_modules)

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        curr_w = params
        x_input = x

        for i, module in enumerate(self._inr_modules):
            module_params = curr_w[:, :module.num_external_params]

            if 0 < i < len(self._inr_modules) - 1 and isinstance(module, INRLinear):
                x = torch.cat([x, x_input], dim=1) # [num_inrs, dim * 2, num_coords]

            x = module(x, module_params)
            curr_w = curr_w[:, module.num_external_params:]

        return x

    def __repr__(self) -> str:
        return f'INRInputSkip {len(self)}'


class INRCoordsSkip(INRModule):
    """
    INR architecture which adds a skip-connection
    of input coordinates to each subsequent layer
    """
    def __init__(self, *modules, concat_to_the_first: bool=True):
        super(INRCoordsSkip, self).__init__()

        self._inr_modules = nn.ModuleList(modules)
        self.concat_to_the_first = concat_to_the_first
        self.num_external_params = sum(c.num_external_params for c in self._inr_modules)

    def __len__(self) -> int:
        return len(self._inr_modules)

    def forward(self, x: Tensor, coord_feats: Tensor, params: Tensor) -> Tensor:
        curr_w = params

        for i, module in enumerate(self._inr_modules):
            module_params = curr_w[:, :module.num_external_params]

            if (i > 0 or self.concat_to_the_first) and isinstance(module, INRLinear):
                x = torch.cat([x, coord_feats], dim=1) # [num_inrs, x_dim + coords_feat_dim, num_coords]

            x = module(x, module_params)
            curr_w = curr_w[:, module.num_external_params:]

        return x

    def __repr__(self) -> str:
        return f'INRCoordsSkip {len(self)}'


class MultiModalINRFactorizedSELinear(INRFactorizedSELinear):
    def __init__(self, in_features: int, out_features: int, weight_std: float, bias_std: float,
        rank: int, num_modes: int, temperature: float, init_dist: str="normal"):
        INRModule.__init__(self)

        self.rank = rank
        self.num_modes = num_modes
        self.in_features = in_features
        self.out_features = out_features
        self.weight_std = weight_std
        self.bias_std = bias_std
        self.temperature = temperature

        self.weights = nn.Parameter(torch.empty(self.num_modes, out_features, in_features))

        # Though self.weights is 3D, torch works fine with it.
        initialize_weights_(self.weights, self.weight_std, init_dist)

        # mode_softmax + lhs + rhs + bias
        self.num_external_params = self.num_modes + rank * in_features + rank * out_features + out_features

    def transform_params(self, params: Tensor) -> Tensor:
        num_inrs = len(params)
        weight_modulation, bias = self.process_external_params(params[:, self.num_modes:])
        modes_probs = (params[:, :self.num_modes] / self.temperature).softmax(dim=1) # [num_inrs, num_modes]
        modes_probs = modes_probs.view(num_inrs, self.num_modes, 1, 1) # [num_inrs, num_modes, 1, 1]
        W_shared = self.weights
        # W_shared = self.weight_std * self.weights
        weight = (W_shared.unsqueeze(0) * modes_probs).sum(dim=1) # [num_inrs, out_features, in_features]
        weight = weight * weight_modulation

        return weight, bias

    def __repr__(self) -> str:
        return f'MultiModalINRFactorizedSELinear {self.weights.shape[2]} -> {self.weights.shape[1]} ({self.num_modes} modes)'


class MultiModalINRSharedLinear(INRFactorizedSELinear):
    def __init__(self, in_features: int, out_features: int, weight_std: float, bias_std: float, num_modes: int, temperature: float):
        INRModule.__init__(self)

        self.num_modes = num_modes
        self.in_features = in_features
        self.out_features = out_features
        self.weight_std = weight_std
        self.bias_std = bias_std
        self.temperature = temperature

        self.weights = nn.Parameter(torch.empty(self.num_modes, out_features, in_features))
        initialize_weights_(self.weights, 1.0, init_dist)

        # mode_softmax + bias
        self.num_external_params = self.num_modes + out_features

    def transform_params(self, params: Tensor) -> Tensor:
        num_inrs = len(params)
        modes_probs = (params[:, :self.num_modes] / self.temperature).softmax(dim=1) # [num_inrs, num_modes]
        modes_probs = modes_probs.view(num_inrs, self.num_modes, 1, 1) # [num_inrs, num_modes, 1, 1]
        W_shared = self.weight_std * self.weights
        weight = (W_shared.unsqueeze(0) * modes_probs).sum(dim=1) # [num_inrs, out_features, in_features]
        bias = params[:, self.num_modes:].view(num_inrs, self.out_features, 1) * self.bias_std

        return weight, bias

    def __repr__(self) -> str:
        return f'MultiModalINRSharedLinear {self.weights.shape[2]} -> {self.weights.shape[1]} ({self.num_modes} modes)'


class INRSVDLinear(INRLinear):
    def __init__(self, in_features: int, out_features: int, weight_std: float, bias_std: float):
        super(INRSVDLinear, self).__init__(in_features, out_features, weight_std=weight_std, bias_std=bias_std)

        weight = torch.empty(out_features, in_features)
        bound = weight_std * np.sqrt(3)
        nn.init.uniform_(weight, -bound, bound)

        U, _, V = torch.svd(weight)
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)

        self.in_features = in_features
        self.out_features = out_features
        self.num_singular_values = min(in_features, out_features)
        self.num_external_params = self.num_singular_values + out_features # num_singular_values + bias_size

    def transform_params(self, params: Tensor) -> Tensor:
        num_inrs = len(params)

        singular_values = params[:, :self.num_singular_values].sigmoid() * 2 # [num_inrs, num_singular_values]
        biases = params[:, self.num_singular_values:] # [num_inrs, out_features]

        V_fused = singular_values.unsqueeze(2) * self.V.view(1, self.num_singular_values, self.in_features) # [num_inrs, num_singular_values, in_features]
        weight = self.U @ V_fused # [num_inrs, out_features, in_features]
        bias = biases.view(num_inrs, self.out_features, 1) * self.bias_std

        return weight, bias

    def __repr__(self) -> str:
        return f'INRSVDLinear {self.V.shape[1]} -> {self.U.shape[0]}'


class INRResConnector(INRModule):
    """
    It is a block from hierarchical INR which operates on a single resolution
    """
    def __init__(self, lr_feat_dim: int, hr_coord_feat_dim: int, output_dim: int, upsampling_mode: str, **factorized_se_kwargs):
        super(INRResConnector, self).__init__()

        self.lr_feat_dim = lr_feat_dim
        self.hr_coord_feat_dim = hr_coord_feat_dim
        self.output_dim = output_dim
        self.upsampling_mode = upsampling_mode

        lr_feats_scale = np.sqrt(1 / lr_feat_dim)
        hr_coords_scale = np.sqrt(1 / hr_coord_feat_dim)

        self.lr_feat_embedder = INRFactorizedSELinear(
            lr_feat_dim, output_dim, weight_std=lr_feats_scale,
            bias_std=lr_feats_scale, **factorized_se_kwargs)
        self.hr_coord_embedder = INRFactorizedSELinear(
            hr_coord_feat_dim, output_dim, weight_std=hr_coords_scale,
            bias_std=hr_coords_scale, **factorized_se_kwargs)

        self.num_external_params = self.lr_feat_embedder.num_external_params + self.hr_coord_embedder.num_external_params

    def forward(self, lowres_feats: Tensor, hr_coord_feats: Tensor, params: Tensor) -> Tensor:
        """
        params:
        - lowres_feats: input data of size [num_inrs, lr_hid_dim, num_lr_coords]
        - hr_coord_feats: highres coordinates features for the current resolution of size [num_inrs, pos_emb_dim, num_lr_coords * 4]
        - params: external params of size [num_inrs, num_external_params]
        """
        num_inrs = len(params)
        num_lr_coords = lowres_feats.shape[2]
        num_hr_coords = hr_coord_feats.shape[2]
        lr_img_size = int(np.sqrt(num_lr_coords))
        hr_img_size = int(np.sqrt(num_hr_coords))
        upscale_factor = hr_img_size // lr_img_size

        assert (lr_img_size ** 2) == num_lr_coords, f"Wrong shape: {lowres_feats.shape}"
        assert params.shape[1] == self.num_external_params, f"Wrong params shape: {params.shape}"

        lowres_context = self.lr_feat_embedder(lowres_feats, params[:, :self.lr_feat_embedder.num_external_params]) # [num_inrs, output_dim, num_lr_coords]
        hr_coords_embs = self.hr_coord_embedder(hr_coord_feats, params[:, self.lr_feat_embedder.num_external_params:]) # [num_inrs, output_dim, num_hr_coords]

        # Repeat low-res feats for each high-resolution coordinate
        lowres_context = lowres_context.view(num_inrs, self.output_dim, lr_img_size, lr_img_size) # [num_inrs, output_dim, lr_img_size, lr_img_size]
        lowres_context = F.interpolate(lowres_context, size=(hr_img_size, hr_img_size), mode=self.upsampling_mode) # [num_inrs, output_dim, hr_img_size, hr_img_size]
        # upsample_factor = hr_img_size // lr_img_size
        # lowres_context = blur_upsample(lowres_context, [1, 3, 3, 1], upsample_factor) # [num_inrs, output_dim, hr_img_size, hr_img_size]
        lowres_context = lowres_context.view(num_inrs, self.output_dim, num_hr_coords) # [num_inrs, output_dim, num_hr_coords]

        # Sum up the results. This is equivalent to concatenation.
        result = lowres_context + hr_coords_embs

        return result

    def __repr__(self) -> str:
        return f'INRResConnector (dim: ) [{self.lr_feat_dim} + {self.hr_coord_feat_dim}] -> {self.output_dim}'


class INRPixelNorm(INRModule):
    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        """
        params:
        - x: input data of size [num_inrs, hid_dim, num_coords]
        - params: external params of size [num_inrs, num_external_params]
        """
        return (x / (x.norm(dim=1, keepdim=True) + 1e-8)) * np.sqrt(x.shape[1])



class INRToRGB(INRLinear):
    def __init__(self, input_dim: int, output_activation: Optional[str], upsampling_mode: str, *args, **kwargs):
        super(INRToRGB, self).__init__(input_dim, 3, *args, **kwargs)
        self.output_activation = output_activation
        self.upsampling_mode = upsampling_mode

    def forward(self, x: Tensor, skip: Optional[Tensor], params: Tensor):
        """
        params:
        - x: input data of size [num_inrs, hid_dim, num_coords]
        - skip: images of size [num_inrs, 3, lr_img_size, lr_img_size]
        - params: external params of size [num_inrs, num_external_params]
        """
        num_inrs = x.shape[0]
        hr_img_size = int(np.sqrt(x.shape[2]))
        assert hr_img_size ** 2 == x.shape[2], f"Wrong shape: {x.shape}"

        out = INRLinear.forward(self, x, params)
        out = out.view(num_inrs, 3, hr_img_size, hr_img_size)

        if self.output_activation == 'pixel_norm':
            out = out / out.detach().norm(dim=1, keepdim=True) * np.sqrt(3)
        elif self.output_activation == 'tanh':
            out = out.tanh()
        elif self.output_activation in ['none', None]:
            pass
        else:
            raise NotImplementedError(f'Unknown output activation: {self.output_activation}')

        if not skip is None:
            upsample_factor = hr_img_size // skip.shape[2]
            skip = F.interpolate(skip, size=(hr_img_size, hr_img_size), mode=self.upsampling_mode)
            # skip = blur_upsample(skip, [1, 3, 3, 1], upsample_factor)
            out = out + skip

        return out


class INRNoiseInjection(INRModule):
    def __init__(self):
        super().__init__()

        self.noise_scale = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: Tensor, params: Tensor, noise=None):
        """
        params:
        - x: input data of size [num_inrs, hid_dim, num_coords]
        - params: external params of size [num_inrs, num_external_params]
        - noise: external noise of size [num_inrs, 1, num_coords]
        """
        if noise is None:
            num_inrs, _, num_coords = x.shape
            noise = x.new_empty(num_inrs, 1, num_coords).normal_()

        return x + self.noise_scale * noise

    def __repr__(self) -> str:
        return f"INRNoiseInjection ({self.noise_scale.item()})"


class INRFourierFeats(INRModule):
    """
    Tired of using random fourier positional embeddings? This module is just for you!
    Assumes that number of input coods is 2
    """
    def __init__(self, config: Config):
        super().__init__()

        # Each period has 4 components: up, down, down, up (for sine)
        # We have the following freqs: [up], [up, down], [up, down, down, up], ...
        # This means number of periods: 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, ...
        # If each period spans 4 pixels at minimum and 4 * img_size at maximum
        # then there is no purpose in having more features then log2(img_size) + 1
        self.config = config

        if self.config.max_num_fixed_coord_feats > 0:
            self.generate_fixed_feats()

        self.num_external_params = self.config.num_learnable_coord_feats * 2

    def get_num_feats(self) -> int:
        result = self.config.num_learnable_coord_feats

        if self.config.max_num_fixed_coord_feats > 0:
            result += self.basis.shape[0]

        return result

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        """
        params:
        - x: input data of size [num_inrs, 2, num_coords]
        - params: external params of size [num_inrs, num_external_params]
        """
        self._check_params_size(params), f"Wrong shape: {params.shape}"

        num_inrs = params.shape[0]
        learnable_basis = self.config.fourier_scale * params.view(num_inrs, self.config.num_learnable_coord_feats, 2)

        if self.config.max_num_fixed_coord_feats > 0:
            fixed_basis = self.basis.unsqueeze(0).repeat(num_inrs, 1, 1) # [num_inrs, fixed_basis_len, 2]
            basis = torch.cat([learnable_basis, fixed_basis], dim=1) # [num_inrs, fixed_basis_len + extern_basis_len, 2]
        else:
            basis = learnable_basis

        x = x.permute(0, 2, 1) # [num_inrs, num_coords, 2]
        y = x @ basis.permute(0, 2, 1) # [num_inrs, num_coords, num_feats]
        y = y.permute(0, 2, 1) # [num_inrs, num_feats, num_coords]

        return y

    def generate_fixed_feats(self) -> Tensor:
        num_feats_per_direction = np.ceil(np.log2(self.config.resolution)).astype(int) + 1
        bases = [
            self.generate_horizontal_basis(num_feats_per_direction),
            self.generate_vertical_basis(num_feats_per_direction),
            self.generate_diag_main_basis(num_feats_per_direction),
            self.generate_anti_diag_basis(num_feats_per_direction),
        ]

        # First, trying to remove diag features
        if num_feats_per_direction * len(bases) > self.config.max_num_fixed_coord_feats:
            if not self.config.use_diag_feats:
                bases = bases[:2]

        # Then, removing extra features...
        if num_feats_per_direction * len(bases) > self.config.max_num_fixed_coord_feats:
            num_exceeding_feats = (num_feats_per_direction * len(bases) - self.config.max_num_fixed_coord_feats) // len(bases)

            if self.config.get('select_high_freq_first'):
                bases = [b[num_exceeding_feats:] for b in bases]
            else:
                bases = [b[:-num_exceeding_feats] for b in bases]

        self.register_buffer('basis', torch.cat(bases, dim=0))

        assert self.basis.shape[0] <= self.config.max_num_fixed_coord_feats, \
            f"num_coord_feats > max_num_fixed_coord_feats: {self.basis.shape, self.config.max_num_fixed_coord_feats}"

    def generate_horizontal_basis(self, num_feats: int) -> Tensor:
        return self.generate_basis(num_feats, [0.0, 1.0], 4.0)

    def generate_vertical_basis(self, num_feats: int) -> Tensor:
        return self.generate_basis(num_feats, [1.0, 0.0], 4.0)

    def generate_diag_main_basis(self, num_feats: int) -> Tensor:
        return self.generate_basis(num_feats, [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))

    def generate_anti_diag_basis(self, num_feats: int) -> Tensor:
        return self.generate_basis(num_feats, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))

    def generate_basis(self, num_feats: int, basis_block: List[float], period_length: float) -> Tensor:
        period_coef = 2.0 * np.pi / period_length
        basis = torch.tensor([basis_block]).repeat(num_feats, 1) # [num_feats, 2]
        powers = torch.tensor([2]).repeat(num_feats).pow(torch.arange(num_feats)).unsqueeze(1) # [num_feats, 1]
        result = basis * powers * period_coef # [num_feats, 2]

        return result.float()

    def __repr__(self):
        return f"INRFourierFeats (shape={self.basis.shape})"



def initialize_weights_(weights: Tensor, std: float, dist_type: str):
    if dist_type == 'normal':
        nn.init.normal_(weights, mean=0.0, std=std)
    elif dist_type == 'uniform':
        bound = std * np.sqrt(3)
        nn.init.uniform_(weights, -bound, bound)
    else:
        raise NotImplementedError(f"Unkown dist type: {dist_type}")
