import random
import torch
from torch import nn
from torch import Tensor

class RandomAug(nn.Module):
    def __init__(self, hid_dim: int):
        super().__init__()

        # self.aug = nn.Sequential(
        #     nn.Conv2d(3, hid_dim, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(hid_dim, 3, 3, padding=1),
        # )
        self.aug = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, img: Tensor) -> Tensor:
        return self.aug(img)


class RandomAugList(nn.Module):
    """
    Randomly chooses between random augs
    """
    def __init__(self, hid_dim: int, num_augs: int):
        super().__init__()

        self.augs = nn.ModuleList([RandomAug(hid_dim) for _ in range(num_augs)])

    def forward(self, img: Tensor) -> Tensor:
        batch_size = img.shape[0]
        allocation = random.sample(range(len(self.augs)), batch_size)
        # TODO(universome): for large batch sizes, this is very slow...
        out = torch.cat([self.augs[allocation[i]](x.unsqueeze(0)) for i, x in enumerate(img)], dim=0)

        return out
