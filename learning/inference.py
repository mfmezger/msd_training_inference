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
    for epoch in range(epochs):
        loss = 0
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)

            
            # compute reconstructions
            outputs = model(x)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, y)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
    
    # compute the epoch training loss
    loss = loss / len(train_loader)
    
    # display the epoch training loss
    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))


    # test the model.


    # find a fitting threshold.






if __name__ == "__main__":
    main()
