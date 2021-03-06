import pandas
import torch
from DataFrameDataset import DataFrameDataSet
from autoencoder import AutoEncoder
from monai.networks.nets import UNet
from PTTestDataSet import TorchTestDataSet


def main():
    # load the training config.
    with open("training.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # initialise test dataset.
    data_dir_val = cfg["data"]["data_dir_val"]
    test_dataset = TorchTestDataSet(directory=data_dir_val)
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initalize the model
    model = UNet().to(device)

    # start the training.
    for x, name in test_loader:
        x = x.to(device)

        # predict segmentation.
        outputs = model(x)

        # save the outputs to pt files.
        torch.save({"vol": outputs, "name": name}, "./outputs/" + name)


if __name__ == "__main__":
    main()
