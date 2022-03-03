# install requirements
python3 pip install -r requirements.txt

# enable wandb logging.


# start the download and extraction. 
python3 preprocessing/download_and_extract.py

# start the training.
python learning/training.py

# start the inference.
python learning/inference.py


