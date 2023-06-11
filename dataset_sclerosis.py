import os
import pathlib

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from PIL import Image


data_transform = transforms.Compose([
    # Resize the images to 128x128
    transforms.Resize(size=(128, 128)),
    # Turn the image into a torch.Tensor
    transforms.ToTensor()  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])


# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir: str, transform=None):
        self.paths = list(
            pathlib.Path(targ_dir).glob("*/*.png"))
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = '', ''

    def load_image(self, index: int):
        """Opens an image via a path and returns it."""
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Returns one sample of data, data and label (X, y)."""
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx  # return data, label (X, y)
        else:
            return img, class_idx  # return data, label (X, y)
