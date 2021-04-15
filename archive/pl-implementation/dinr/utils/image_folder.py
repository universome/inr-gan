from typing import Dict
from torchvision.datasets import ImageFolder as TVImageFolder


class ImageFolder(TVImageFolder):
    """
    It's like ImageFolder class from torchvision, but a different API
    """
    def __getitem__(self, idx: int) -> Dict:
        image, label = TVImageFolder.__getitem__(self, idx)

        return {"img": image, "label": label}
