import os
from typing import Optional, Callable

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader


class PlainImageDataset(VisionDataset):
    def __init__(self, root: os.PathLike, transform: Optional[Callable]=None):
        self.root = root
        self.transform = transform
        self.imgs_paths = [os.path.join(self.root, p) for p in os.listdir(self.root)]

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx: int):
        image = pil_loader(self.imgs_paths[idx])

        if not self.transform is None:
            image = self.transform(image)

        return {"img": image}
