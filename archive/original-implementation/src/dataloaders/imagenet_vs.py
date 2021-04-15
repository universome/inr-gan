import os
from typing import Tuple, List, Any, Iterable

import lmdb
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torch.utils.data import Dataset
from torch import Tensor


class VariableSizedImageNet(Dataset):
    """
    This dataset loads variable-sized ImageNet 128
    and pads it with black pixels
    """
    def __init__(self, root_path, transform):
        class_names = sorted(os.listdir(root_path))
        filenames = [os.listdir(os.path.join(root_path, c)) for c in class_names]

        self.labels = [c for c, fs in enumerate(filenames) for _ in fs]
        self.filepaths = [os.path.join(root_path, class_names[self.labels[i]], f) for i, f in enumerate(flatten(filenames))]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        img = Image.open(self.filepaths[index])
        img = self.transform(img)

        return (img, self.labels[index])


def aspect_ratio_to_wh(aspect_ratio: float, larger_side_target_size: int) -> Tuple[int, int]:
    """
    Assuming aspect ratio to be in W/H format
    Returns w, h
    """
    if aspect_ratio == 1.0:
        h = w = larger_side_target_size
    elif aspect_ratio < 1.0:
        h = larger_side_target_size
        w = round(h * aspect_ratio)
    else:
        w = larger_side_target_size
        h = round(w / aspect_ratio)

    return (w, h)


def crop_and_fill_by_aspect_ratio(img: Tensor, aspect_ratio: float, fill_value: float=0.0) -> Tensor:
    """
    Filling the square image borders determined by the aspect ratio
    Image is assumed to be a CHW tensor and aspect ratio is in W/H format
    """
    assert img.shape[0] == 3, f"Assumed to be CHW: {img.shape}"
    assert img.ndim == 3, f"Assumed to have 3 dims: {img.shape}"
    assert img.shape[1] == img.shape[2], f"Assumed to be a square: {img.shape}"

    if aspect_ratio == 1.0:
        return img # Do nothing

    side_size = img.shape[1]
    target_w, target_h = aspect_ratio_to_wh(aspect_ratio, side_size)
    padding = compute_pad_to_square_size(target_w, target_h)
    # Convert padding to borders to avoid [0:-0] problem
    borders = (padding[0], padding[1], side_size - padding[2], side_size - padding[3])
    result = torch.empty_like(img).fill_(fill_value)
    result[:, borders[1]:borders[3], borders[0]:borders[2]] = img[:, borders[1]:borders[3], borders[0]:borders[2]]

    return result


def resize_and_fill_by_aspect_ratio(img: Tensor, aspect_ratio: float, fill_value: float=0.0) -> Tensor:
    """
    Filling the square image borders determined by the aspect ratio
    Image is assumed to be a CHW tensor and aspect ratio is in W/H format
    """
    assert img.shape[0] == 3, f"Assumed to be CHW: {img.shape}"
    assert img.ndim == 3, f"Assumed to have 3 dims: {img.shape}"
    assert img.shape[1] == img.shape[2], f"Assumed to be a square: {img.shape}"

    if aspect_ratio == 1.0:
        return img # Do nothing

    side_size = img.shape[1]
    target_w, target_h = aspect_ratio_to_wh(aspect_ratio, side_size)

    # Resizing
    img_resized = F.interpolate(img.unsqueeze(0), (target_h, target_w), mode='bilinear').squeeze(0)

    # Filling procedure is a bit cumbersome
    # We do this by inserting img_resized into a black square
    padding = compute_pad_to_square_size(target_w, target_h)
    # Convert padding to borders to avoid [0:-0] problem
    borders = (padding[0], padding[1], side_size - padding[2], side_size - padding[3])
    result = torch.empty_like(img).fill_(fill_value)
    result[:, borders[1]:borders[3], borders[0]:borders[2]] = img_resized

    return result


def lu_crop_and_fill_by_aspect_ratio(img: Tensor, aspect_ratio: float, fill_value: float=0.0) -> Tensor:
    """
    "Left-upper" crop of a given image. Instead of slicing the borders,
    we slice right/bottom part of it that overflows a given aspect ratio
    Image is assumed to be a CHW tensor and aspect ratio is in W/H format
    """
    assert img.shape[0] == 3, f"Assumed to be CHW: {img.shape}"
    assert img.ndim == 3, f"Assumed to have 3 dims: {img.shape}"
    assert img.shape[1] == img.shape[2], f"Assumed to be a square: {img.shape}"

    if aspect_ratio == 1.0:
        return img # Do nothing

    side_size = img.shape[1]
    target_w, target_h = aspect_ratio_to_wh(aspect_ratio, side_size)
    padding = compute_pad_to_square_size(target_w, target_h)
    n_pixels_w = side_size - (padding[0] + padding[2])
    n_pixels_h = side_size - (padding[1] + padding[3])
    result = torch.empty_like(img).fill_(fill_value)
    result[:, :n_pixels_h, :n_pixels_w] = img[:, :n_pixels_h, :n_pixels_w]

    return result


def fill_by_aspect_ratio(images: Iterable[Tensor], aspect_ratios: Iterable[float], resize_strategy: str, fill_value: float=0.0) -> Tensor:
    if resize_strategy == 'crop':
        images = [crop_and_fill_by_aspect_ratio(x, a, fill_value) for (x, a) in zip(images, aspect_ratios)]
    elif resize_strategy == 'resize':
        images = [resize_and_fill_by_aspect_ratio(x, a, fill_value) for (x, a) in zip(images, aspect_ratios)]
    elif resize_strategy == 'lu_crop':
        images = [lu_crop_and_fill_by_aspect_ratio(x, a, fill_value) for (x, a) in zip(images, aspect_ratios)]
    else:
        raise NotImplementedError(f"Unknown resize strategy: {resize_strategy}")

    return torch.stack(images, dim=0)


def compute_pad_to_square_size(w, h) -> Tuple[int, int, int, int]:
    """
    Computes padding ammount for an image to become a square
    Returns: (pad_left, pad_top, pad_right, pad_bottom) tuple
    """
    diff = abs(h - w)

    if diff == 0:
        return (0, 0, 0, 0)

    if diff % 2 == 0:
        one_side_pad = diff // 2
        other_side_pad = diff // 2
    else:
        one_side_pad = diff // 2
        other_side_pad = diff // 2 + 1

    if h > w:
        padding = (one_side_pad, 0, other_side_pad, 0)
    elif h < w:
        padding = (0, one_side_pad, 0, other_side_pad)
    else:
        assert False

    return padding


def compute_oneside_pad_to_square_size(w, h) -> Tuple[int, int, int, int]:
    diff = abs(h - w)

    if diff == 0:
        return (0, 0, 0, 0)

    if h > w:
        padding = (0, 0, diff, 0)
    elif h < w:
        padding = (0, 0, 0, diff)
    else:
        assert False

    return padding


class PadToSquare:
    def __init__(self, pad_strategy: str='two_sided', **pad_kwargs):
        self.pad_strategy = pad_strategy
        self.pad_kwargs = pad_kwargs

    def __call__(self, img: Image) -> Image:
        w, h = img.size[0], img.size[1]

        if w == h:
            return img

        if self.pad_strategy == 'two_sided':
            padding = compute_pad_to_square_size(w, h)
        elif self.pad_strategy == 'one_sided':
            padding = compute_oneside_pad_to_square_size(w, h)
        else:
            raise NotImplementedError

        padded = TVF.pad(img, padding, **self.pad_kwargs)

        return padded

def flatten(x: List[List[Any]]) -> List[Any]:
    return [z for y in x for z in y]
