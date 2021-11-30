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

    # define DataSet and DataLoader.

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initalize the model
    model = UNet().to(device)

    # start the training.
    # make epoch dependable on dataloader
    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, 784).to(device)
            
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(batch_features)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)
            
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
