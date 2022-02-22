# README


## Data Source

Medical Segmentation Decathlon http://medicaldecathlon.com/

## TODOS

 - [ ] Z Normalisation
 - [ ] Add more metrics
 - [ ] Test on server.
 - [ ] Inference Script
 - [ ] WandB and Logging
 - [ ] Training routine regarding imaging size and so on.


## Installation

Create an Anaconda environment 'conda create -n msd python=3.8'.  
Run the requirments.txt file. 'pip install -r requirements.txt'

## Usage

### Preprocessing
The script 'download_extract.py' in the preprocessing folder downloads the Medical Segmentation Dataset from the AWS Open Data Repository. You can change between US and Europe AWS Servers by setting the EU_or_NA variable in the config.yml file.

### Training
The script 'training.py' in the learning folder trains the model. You can change the model type, the number of epochs, the batch size, the learning rate, the number of workers and other variables in the training.yml file.


## Credits
Implementation by Marc Fabian Mezger (@mfmezger) https://github.com/mfmezger


# Fragen für SambaNova
* Data augmentation on GPU possible? -> compile that or inefficient and better done on CPU?
    --> pytorch läuft auf dem system augmentation ist possible.

* is an AWS in NA West okay for data download? -> okay wird gecacht.
* Can we have access to the DOCs or should i just leave my code for gpu and SambaNova puts their model in? -> bei bedarf.
* can we log to Weights and Biases during training or only offline logging? -> 
* how many runs can we do?
* 

checkpointing noch integrieren.

loss direkt in forward loop berechnen. init pass für größe des models.

get imputs methode mit samba variablen samba.randn
nur eine größe als input möglich.

samba from torch & to_torch.

beispiel github für 3d und 2d mit daten.

demo script für samba nova mit data download, umwandeln, trainieren. 3D Unet

3D Unet als 2x 2D Unet verwenden. --> Samba Nova fragen.
Patching 3D volume als 2D Bild anneinander gehängt speichern.

