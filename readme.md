# README Medical Segmentation Decathlon Download/Extraction, Training, Inference Script

## Data Source

Medical Segmentation Decathlon http://medicaldecathlon.com/

## TODOS

- [ ] Test on linux pc.
- [ ] Loop over the datasets.
-  sambanova switch --> torch randn
 pc build vorschlag.

## Installation

Create an Anaconda environment 'conda create -n msd python=3.8'.  
Run the requirments.txt file. 'pip install -r requirements.txt'

## Usage

As described in the server.sh the following commands have to be executed in the right order.

First install the requirements. `python3 pip install -r requirements.txt`
Then downloading of the Dataset `python3 preprocessing/download_and_extract.py`  
(Logging into wandb: `wandb login [Your_KEY]` if you want to log to Weights and Biases.  
Starting the Training: `python3 learning/training.py`  
Starting the Inference: `python3 learning/inference.py`

### Preprocessing

The script 'download_extract.py' in the preprocessing folder downloads the Medical Segmentation Dataset from the AWS
Open Data Repository. You can change between US and Europe AWS Servers by setting the EU_or_NA variable in the
config.yml file.

### Training

The script 'training.py' in the learning folder trains the model. You can change the model type, the number of epochs,
the batch size, the learning rate, the number of workers and other variables in the training.yml file.

## Credits

Implementation by Marc Fabian Mezger (@mfmezger) https://github.com/mfmezger
