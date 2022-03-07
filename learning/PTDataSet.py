import os
import numpy as np
import torch
from batchviewer import view_batch

import torch.nn as nn
from torch.utils.data import Dataset


def padding(img, mask, target_size, padding=False, upsample=False, downsample=False):
    """If padding True then pad the image to the target size."""
    if tuple(img.size()) == target_size:
        return img, mask
    print(img.size(), target_size)
    if padding:
        # performance wise probably better. But avoids the issue of the not even pads.
        tmp_img = torch.zeros(target_size)
        tmp_img[:, :, : img.shape[2], : img.shape[3], : img.shape[4]] = img
        img = tmp_img

        tmp_mask = torch.zeros(target_size)
        tmp_mask[:, :, : mask.shape[2], : mask.shape[3], : mask.shape[4]] = mask
        mask = tmp_mask

    if upsample:
        upsample = nn.Upsample(
            size=target_size[2:], mode="trilinear", align_corners=True
        )
        img = upsample(img)
        upsample = nn.Upsample(size=target_size[2:], mode="nearest")
        mask = upsample(mask)

    if downsample:
        img = nn.functional.interpolate(img, size=target_size[2:], mode="trilinear")
        mask = nn.functional.interpolate(mask, size=target_size[2:], mode="nearest")

    return img, mask


class TorchDataSet(Dataset):
    """
    Loading the Datasets
    """

    def __init__(self, directory, padding_bool=False, target_size=(1, 1, 5, 50, 50)):
        self.directory = directory
        self.images = os.listdir(directory)
        self.target_size = target_size
        self.padding_bool = padding_bool

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load the image and the groundtruth
        name = self.images[idx]
        file = torch.load(os.path.join(self.directory, name))

        image = file["vol"]
        mask = file["mask"]

        # change the datatype to float32 if you do not use FP16.
        image = image.to(torch.float32).unsqueeze(0).unsqueeze(0)
        mask = mask.to(torch.float32).unsqueeze(0).unsqueeze(0)
        image_size = tuple(image.size())
        if image_size != self.target_size:
            if self.padding_bool:
                image, mask = padding(image, mask, self.target_size, padding=True)
            else:
                if image_size < self.target_size:
                    image, mask = padding(
                        image,
                        mask,
                        target_size=self.target_size,
                        padding=False,
                        upsample=True,
                    )
                elif image_size > self.target_size:
                    image, mask = padding(
                        image,
                        mask,
                        target_size=self.target_size,
                        padding=False,
                        downsample=True,
                    )

        return image, mask


if __name__ == "__main__":
    # create empty tensor for testing.
    a = torch.randn(10, 100, 100)
    b = torch.randn(10, 100, 100)

    # save the tensor to disk.
    torch.save(
        {"vol": a, "mask": b}, "data\\test.pt",
    )

    # dataset = TorchDataSet(
    #     directory="data/"
    # )
    # img, mask = dataset[0]
    # print(img.shape)

    dataset = TorchDataSet(directory="data/", padding_bool=False,)
    img, mask = dataset[0]
    print(img.shape, mask.shape)

    view_batch(img[0], mask[0], height=512, width=512)
