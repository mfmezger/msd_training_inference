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
 - [ ] Checkpointing

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


Notes
loss direkt in forward loop berechnen. init pass für größe des models.

get imputs methode mit samba variablen samba.randn
nur eine größe als input möglich.

