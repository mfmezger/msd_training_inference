import pandas
import torch
from DataFrameDataset import DataFrameDataSet
from autoencoder import AutoEncoder
from monai.networks.nets import UNet

# hyperparameters
batch_size = 512
epochs = 2000
learning_rate = 1e-3


def main():
    # read the data.

    # todo: create testing dataset.

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initalize the model
    model = UNet().to(device)

    # start the training.
    # make epoch dependable on dataloader
    for x, name in test_loader:
        x = x.to(device)

        # predict segmentation.
        outputs = model(x)

        # save the outputs to pt files.
        torch.save({"vol": outputs, "name": name}, "./outputs/" + name)


if __name__ == "__main__":
    main()
