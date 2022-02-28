import numpy as np
import os
import torch
from batchviewer import view_batch

import torch.nn as nn
from torch.utils.data import Dataset

def padding(img, mask,target_size, padding=False, upsample=False, downsample=False):
    """If padding True then pad the image to the target size."""
    # if img.shape() == target_size:
        # return img, mask

    if padding:
        print(target_size[-2 ], img.shape[-2])
        padding_x = int((target_size[-1] - img.shape[-1])/2)
        padding_y = int((target_size[-2] - img.shape[-2])/2)
        padding_z = int((target_size[-3] - img.shape[-3])/2)
        print(padding_x, padding_y, padding_z)

        img = nn.functional.pad(input=img, pad=(2, padding_z, padding_x, padding_z), mode='constant', value=0)
        mask = nn.functional.pad(input=mask, pad=(2, padding_z, padding_x, padding_y), mode='constant', value=0)
    if upsample:
        #(x, mode=self.mode, scale_factor=self.size)
        img = nn.functional.interpolate(img, [img.shape[-1]*5, img.shape[-1]*5], mode='bilinear')
        mask = nn.functional.interpolate(mask, [img.shape[-1]*5.12, img.shape[-1]*5], mode='bilinear')

        # TODO: resize to nearest possible res than pad.

        # mask = nn.functional.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)
    if downsample:
        img = nn.functional.interpolate(img, size=target_size, mode='bilinear', align_corners=False)
        mask = nn.functional.interpolate(mask, size=target_size, mode='nearest', align_corners=False)

    return img, mask


class TorchDataSet(Dataset):
    """
    Loading the Datasets
    """

    def __init__(self, directory, change_res=False):
        self.directory = directory
        self.images = os.listdir(directory)
        self.change_res = change_res

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

        if self.change_res:
            image, mask = padding(image, mask, target_size=(1,20,512,512), padding=False, upsample=True)

        return image, mask


if __name__ == '__main__':
    # create empty tensor for testing.
    a = torch.randn( 10, 100, 100)
    b = torch.randn( 10, 100, 100)

    # save the tensor to disk.
    torch.save({"vol": a, "mask": b}, "C:\\Users\\Marc\\Documents\\GitHub\\msd_training_inference\\data\\test.pt")

    dataset = TorchDataSet(directory="C:\\Users\\Marc\\Documents\\GitHub\\msd_training_inference\\data\\")
    img, mask = dataset[0]
    print(img.shape)

    # view_batch(img, mask, height=512, width=512)

    dataset = TorchDataSet(directory="C:\\Users\\Marc\\Documents\\GitHub\\msd_training_inference\\data\\", change_res=True)
    img, mask = dataset[0]
    print(img.shape)