import numpy as np
import os
import torch
from batchviewer import view_batch
from scipy.ndimage.filters import gaussian_filter
import torch.nn as nn
from torch.utils.data import Dataset


class TorchDataSet(Dataset):
    """
    Loading the Datasets
    """

    def __init__(self, directory, padding=False):
        self.directory = directory
        self.images = os.listdir(directory)
        self.padding = padding

    def __len__(self):
        return len(self.images)


    def padding(self, target_size, padding):
        """If padding True then pad the image to the target size."""
        if padding:
            self.img = nn.pad(self.img, (0, target_size - self.img.shape[2], 0, target_size - self.img.shape[3]))
            self.mask = nn.pad(self.mask, (0, target_size - self.mask.shape[2], 0, target_size - self.mask.shape[3]))
        if upsample:
            self.img = nn.functional.interpolate(self.img, size=target_size, mode='bilinear', align_corners=False)
            self.mask = nn.functional.interpolate(self.mask, size=target_size, mode='nearest', align_corners=False)
        if downsample:
            self.img = nn.functional.interpolate(self.img, size=target_size, mode='bilinear', align_corners=False)
            self.mask = nn.functional.interpolate(self.mask, size=target_size, mode='nearest', align_corners=False)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load the image and the groundtruth
        name = self.images[idx]
        file = torch.load(os.path.join(self.directory, name))

        image = file["vol"]
        mask = file["mask"]

        # change the datatype to float32 if you do not use FP16.
        image = image.to(torch.float32).unsqueeze(0)
        mask = mask.to(torch.float32).unsqueeze(0)

        return image, mask


if __name__ == '__main__':
    dataset = TorchDataSet(directory="data/")
    img, mask = dataset[0]
    print(img.shape)

    view_batch(img, mask, height=512, width=512)

    # TODO: Implement tests for padding.
