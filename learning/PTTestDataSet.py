import os
import torch
from batchviewer import view_batch

import torch.nn as nn
from torch.utils.data import Dataset


def padding(img, target_size, padding=False, upsample=False, downsample=False):
    """If padding True then pad the image to the target size."""
    if tuple(img.size()) == target_size:
        return img, mask
    print(img.size(), target_size)
    if padding:
        # performance wise probably better. But avoids the issue of the not even pads.
        tmp_img = torch.zeros(target_size)
        tmp_img[:, :, : img.shape[2], : img.shape[3], : img.shape[4]] = img
        img = tmp_img

    if upsample:
        upsample = nn.Upsample(
            size=target_size[2:], mode="trilinear", align_corners=True
        )
        img = upsample(img)

    if downsample:
        img = nn.functional.interpolate(img, size=target_size[2:], mode="trilinear")

    return img


class TorchTestDataSet(Dataset):
    """
    Loading the Datasets
    """

    def __init__(self, directory, target_size=(1, 20, 512, 512), padding_bool=False):
        self.directory = directory
        self.images = os.listdir(directory)
        self.change_res = change_res
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

        # change the datatype to float32 if you do not use FP16.
        image = image.to(torch.float32).unsqueeze(0)

        if image.shape == target_size:
            if padding_bool:
                image = padding(image, target_size, padding=True)
            else:
                if image.shape < target_size:
                    image = padding(
                        image,
                        target_size=self.target_size,
                        padding=False,
                        upsample=True,
                    )
                elif image.shape > target_size:
                    image = padding(
                        image,
                        target_size=self.target_size,
                        padding=False,
                        downsample=True,
                    )

        return image


if __name__ == "__main__":
    # create empty tensor for testing.
    a = torch.randn(10, 100, 100)

    # save the tensor to disk.
    torch.save(
        {"vol": a}, "test.pt",
    )

    dataset = TorchTestDataSet(directory="data/",) 
    img = dataset[0]
    print(img.shape)

    dataset = TorchTestDataSet(directory="data/", change_res=True,)
    img = dataset[0]
    print(img.shape)

    # TODO: load original image as well.
    view_batch(img, height=512, width=512)
