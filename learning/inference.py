import torch
import pandas
from autoencoder import AutoEncoder
from DataFrameDataset import DataFrameDataSet
from monai.networks.nets import UNet

# hyperparameters
batch_size = 512
epochs = 2000
learning_rate = 1e-3


def main():
    # read the data.

    # TODO: define DataSet and DataLoader.
    # create PTDataset.
    

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
