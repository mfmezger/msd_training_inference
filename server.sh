# install requirements
python3 pip install -r requirements.txt


# start the download and extraction. 
python3 preprocessing/download_and_extract.py

# enable wandb logging.
wandb login [Your_KEY]

# start the training.
python3 learning/training.py

# start the inference.
python3 learning/inference.py


