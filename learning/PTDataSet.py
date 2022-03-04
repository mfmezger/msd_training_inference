import os
import numpy as np
import torch
from batchviewer import view_batch

import torch.nn as nn
from torch.utils.data import Dataset


def padding(img, mask, target_size, padding=False, upsample=False, downsample=False):
    """If padding True then pad the image to the target size."""
    if  tuple(img.size()) == target_size:
        return img, mask

    if padding:
        # performance wise probably better. But avoids the issue of the not even pads.
        tmp_img = torch.zeros(target_size)
        tmp_img[0, : img.shape[0], : img.shape[1], : img.shape[2]] = img
        img = tmp_img

        tmp_mask = torch.zeros(target_size)
        tmp_mask[0, : img.shape[0], : img.shape[1], : img.shape[2]] = mask
        mask = tmp_mask

    if upsample:
        upsample = nn.Upsample(size=target_size, mode="bilinear", align_corners=True)
        img = upsample(img)
        upsample = mask.upsample(target_size, mode="nearest")
        mask = upsample(mask)

    if downsample:
        img = nn.functional.interpolate(
            img, size=target_size, mode="bilinear", align_corners=False
        )
        mask = nn.functional.interpolate(
            mask, size=target_size, mode="nearest", align_corners=False
        )

    return img, mask


class TorchDataSet(Dataset):
    """
    Loading the Datasets
    """

    def __init__(self, directory, padding_bool=False, target_size=(1, 20, 512, 512)):
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
        image = image.to(torch.float32).unsqueeze(0)
        mask = mask.to(torch.float32).unsqueeze(0)
        image_size = tuple(image.size())
        if image_size != self.target_size:
            if self.padding_bool:
                image, mask = padding(image, mask, self.target_size, padding=True)
            else:
                if  image_size< self.target_size:
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
        {"vol": a, "mask": b},
        "../data/test.pt",
    )

    # dataset = TorchDataSet(
    #     directory="data/"
    # )
    # img, mask = dataset[0]
    # print(img.shape)


    dataset = TorchDataSet(
        directory="data/",
        padding_bool=True,
    )
    img, mask = dataset[0]
    print(img.shape)

    view_batch(img, mask, height=512, width=512)