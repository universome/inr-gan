import os
from typing import Callable
from torchvision import datasets
from torchvision import transforms
from firelab.config import Config

from src.dataloaders.lsun import LSUN
from src.dataloaders.imagenet_vs import VariableSizedImageNet, PadToSquare
from src.utils.constants import DEBUG
from src.dataloaders.utils import (
    CenterCropToMin,
    PlainImageDataset,
    SingleImageDataset,
    PatchConcatAndResize,
)


def load_data(config: Config, transform: Callable=None):
    transform = get_transform(config) if transform is None else transform

    if config.name == 'mnist':
        dataset = datasets.MNIST(config.dir, train=True, download=True, transform=transform)
    elif config.name == 'cifar10':
        dataset = datasets.CIFAR10(config.dir, train=True, download=True, transform=transform)
    elif config.name.startswith('lsun_'):
        split = 'val' if DEBUG else 'train' # ('val' split is much smaller)
        category_name = config.name[config.name.find('_') + 1:]

        # We have to convert dir path "/.../lsun/bedroom_train_lmdb" => "/.../lsun"
        # because LSUN dataset expects root ds path as input
        ds_root_dir = os.path.dirname(os.path.normpath(config.dir))

        dataset = LSUN(ds_root_dir, classes=[f'{category_name}_{split}'], transform=transform)
    elif config.name in {'ffhq_thumbs', 'celeba_thumbs', 'ffhq_256', 'ffhq_1024'}:
        dataset = PlainImageDataset(config.dir, transform=transform)
    elif config.name == 'single_image':
        dataset = SingleImageDataset(config.img_path, transform=transform)
    elif config.name == 'imagenet_vs':
        dataset = VariableSizedImageNet(config.dir, transform=transform)
    else:
        raise NotImplementedError(f'Unknown dataset: {config.name}')

    return dataset


def get_transform(config: Config) -> Callable:
    if config.name == 'mnist':
        return transforms.Compose([
            transforms.Resize(config.target_img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    elif config.name in {'cifar10', 'single_image'}:
        return transforms.Compose([
            transforms.Resize((config.target_img_size, config.target_img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    elif config.name.startswith('lsun_') or config.name in {'ffhq_thumbs', 'celeba_thumbs', 'ffhq_256', 'ffhq_1024'}:
        if config.get('concat_patches.enabled'):
            return transforms.Compose([
                CenterCropToMin(),
                transforms.RandomHorizontalFlip(),
                PatchConcatAndResize(config.target_img_size, config.concat_patches.ratio),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            return transforms.Compose([
                CenterCropToMin(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(config.target_img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
    elif config.name == 'imagenet_vs':
        return transforms.Compose([
            PadToSquare(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
    else:
        raise NotImplementedError(f'Unknown dataset: {config.name}')
