import torch
import pandas
from monai.networks.nets import UNet
import yaml
from PTDataSet import TorchDataSet
import wandb
from torch.utils.tensorboard import SummaryWriter


def main():
    # read the data.
    # define the data directory
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # hyperparameters from training.yml
    batch_size = cfg["hyper_parameters"]["batch_size"]
    epochs = cfg["hyper_parameters"]["epochs"]
    learning_rate = cfg["hyper_parameters"]["lr"]
    num_workers = cfg["hyper_parameters"]["num_workers"]

    # initialize logging.
    # start with wandb.
    if cfg["logging"]["logging_wandb"]:
        wandb.init(project="msd-SN")
    
    # tensorboard logging.
    if cfg["logging"]["logging_tensorboard"]:
        writer = SummaryWriter()
    
    # csv logging. 
    if cfg["logging"]["logging_csv"]:
        pass

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initalize the model
    model = UNet().to(device)

    # create an optimizer.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # mean-squared error loss
    criterion = nn.CrossEntropyLoss()

    # test initial pass if model capacity is enough.
    # create empty tensor for testing.
    # TODO: find common  size.
    test_input = torch.randn(1, 1, 100, 512, 512)

    # propagate the input through the model
    test_output = model(test_input)

    # test that input shape equals output shape.
    assert test_input.shape == test_output.shape

    # define dataset.
    train_ds = TorchDataSet(directory="data/train/")
    val_ds = TorchDataSet(directory="data/val/")
    # create a dataloader object.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # epoch loops.
    for epoch in range(epochs):
        loss = 0
        val_loss = 0

        # start the training.
        for img, mask in train_loader:
            # load it to the active device
            img = img.to(device)            
            mask = mask.to(device)             
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(img)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, mask)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()


        # validation loop.
        for img, mask in val_loader:
            
            img = img.to(device)            
            mask = mask.to(device)             
            
            # compute reconstructions
            outputs = model(img)
            
            # compute training reconstruction loss
            loss = criterion(outputs, mask)
            
            # TODO: Metrics
            # calculate other metrics. fwIOU. mIou, Dice, etc.


            
            # add the mini-batch training loss to epoch loss
            val_loss += loss.item()



        # logging.
        # wandb logging.
        if cfg["logging"]["logging_wandb"]:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss
            })
        # tensorboard logging.
        if cfg["logging"]["logging_tensorboard"]:
            # log loss
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            # log metrics
        # csv logging.
        if cfg["logging"]["logging_csv"]:
            pass

        # compute the epoch training loss
        loss = loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.8f}".format(epoch + 1, epochs, loss))
        print("epoch : {}/{}, val_loss = {:.8f}".format(epoch + 1, epochs, val_loss))

        # checkpoint the model every 50 epochs.
        if (epoch + 1) % 50 == 0:
            # save the model
            torch.save(model.state_dict(), "model_{}.pth".format(epoch + 1))

    # save the model.
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
