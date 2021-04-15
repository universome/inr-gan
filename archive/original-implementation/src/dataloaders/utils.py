import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader
import torchvision.transforms.functional as TVF
from firelab.utils.training_utils import LinearScheme
from firelab.config import Config
from PIL import Image


class ProgressiveTransform:
    """
    Dataset transforms which can be upgraded during training,
    and this will change the underlying transform function
    """
    def __init__(self, config: Config):
        self.config = config.hp.progressive_transform
        self.target_img_size = config.data.target_img_size

        self.num_iters_done = 0
        self.curr_min_scale = self.config.initial_min_scale
        self.update_freq = self.config.update_freq

        self.min_scale_scheduler = LinearScheme(
            self.config.initial_min_scale,
            self.config.target_min_scale,
            self.config.num_iterations
        )

        self.init_transform_fn()

    def __call__(self, *args, **kwargs):
        return self._transform(*args, **kwargs)

    def update(self, num_iters_done: int):
        self.num_iters_done = num_iters_done

        if self.num_iters_done % self.update_freq == 0:
            # Updating only once per 100 iterations not to slow down things
            self.curr_min_scale = self.min_scale_scheduler.evaluate(self.num_iters_done)
            self.init_transform_fn()

    def create_resize_transform(self) -> Callable:
        if self.config.enabled:
            return transforms.RandomResizedCrop(
                self.target_img_size,
                scale=(self.curr_min_scale, 1.0), ratio=(1.0, 1.0))
        else:
            return transforms.Resize(self.target_img_size)

    def init_transform_fn(self):
        self._transform = transforms.Compose([
            CenterCropToMin(),
            self.create_resize_transform(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])


class CenterCropToMin:
    """
    CenterCrops an image to a min size
    """
    def __call__(self, image):
        assert TVF._is_pil_image(image)

        return TVF.center_crop(image, min(image.size))


class PlainImageDataset(VisionDataset):
    def __init__(self, root: os.PathLike, transform: Optional[Callable]=None):

        self.root = root
        self.transform = transform
        self.imgs_paths = [os.path.join(self.root, p) for p in os.listdir(self.root)]
        self.dummy_class = 0 # TODO: it is a dirty hack so we do not change API compared to LSUN :(

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx: int):
        image = pil_loader(self.imgs_paths[idx])

        if not self.transform is None:
            image = self.transform(image)

        return (image, self.dummy_class)


class SingleImageDataset(VisionDataset):
    """
    Sometimes, for an INR we need to fit a single image
    This class exists so not to deviate from the usual dataloader API
    """
    def __init__(self, img_path: os.PathLike, transform: Optional[Callable]=None):
        self.image = pil_loader(img_path)

        if not transform is None:
            self.image = transform(self.image)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        assert idx == 0

        return self.image


class PatchConcatAndResize:
    """
    Concatenates a random image patch to the main image as an additional channel
    """
    def __init__(self, target_size: int, patch_ratio: float=0.5, interpolation=Image.BILINEAR):
        self.target_size = target_size
        self.patch_ratio = patch_ratio
        self.interpolation = interpolation
        self.rnd = np.random.RandomState(42)

    def get_params(self, height: int, width: int) -> Tuple[int, int, int, int]:
        assert height == width, f"Wrong image shape: {height, width}"

        crop_size = int(self.patch_ratio * width)
        top = self.rnd.randint(0, height - crop_size)
        left = self.rnd.randint(0, width - crop_size)

        return top, left, crop_size, crop_size

    def __call__(self, image: Tensor) -> Tensor:
        top, left, h, w = self.get_params(image.size[0], image.size[1])

        random_crop = TVF.resized_crop(image, top, left, h, w, self.target_size, self.interpolation)
        main_img = TVF.resize(image, self.target_size, self.interpolation)

        return torch.cat([TVF.to_tensor(main_img), TVF.to_tensor(random_crop)], dim=0) # Concatenate along c-dim


def compute_square_padding(height: int, width: int) -> Tuple[int, int, int, int]:
    pad_l, pad_t, pad_r, pad_b = 0, 0, 0, 0

    if width < height:
        diff = height - width

        if diff % 2 == 0:
            pad_l = pad_r = diff // 2
        else:
            pad_l = 1 + diff // 2 # Left pad is 1 pixel bigger
            pad_r = diff // 2
    elif width > height:
        diff = width - height

        if diff % 2 == 0:
            pad_t = pad_b = diff // 2
        else:
            pad_t = 1 + diff // 2 # Top pad is 1 pixel bigger
            pad_b = diff // 2


    return (pad_l, pad_t, pad_r, pad_b)
