import os

import yaml
import nibabel as nib
from monai.apps import download_and_extract
from pathlib import Path
from torchio.transforms import ZNormalization


def preprocessing_ct(image):
    """Preprocess the CT images."""


    pass

def prepocessing_mr(image):
    """Preprocess the MRI images."""

    # do z-score normalization.
    # FIXME: is that really the same as Z Score Normalzation?
    image = ZNormalization(mean=0.0, std=1.0)(image)

    pass

def save_pt(image, label, save_dir):
    """Save the images and labels as pytorch files."""
    pass


def convert_images(cfg, data_dir, save_dir):
    """Convert images to pytorch files."""
    
    # create lists of the images.
    label_list = os.listdir(os.path.join( data_dir, "labelsTr"))
    images_list = os.listdir(os.path.join( data_dir, "imagesTr"))
    image_list_test = os.listdir(os.path.join( data_dir, "imagesTs"))
    
    
    # do the preprocessing based on the category.
    ct_list = ["Task03_Liver", "Task06_Lung", "Task07_Pancreas", "Task10_Colon", "Task08_HepaticVessel", "Task09_Spleen"]
    ct_preprocessing_decision = False
    if data_dir in ct_list:
        ct_preprocessing_decision = True
    
    # load the volumes from images and labels.
    for image, label in zip(images_list, label_list):
        # load the images and labels.
        image_path = os.path.join(data_dir, "imagesTr", image)
        label_path = os.path.join(data_dir, "labelsTr", label)

        # load the images and labels.
        image = nib.load(image_path)
        label = nib.load(label_path)
        
        image = image.get_fdata()
        image = image.transpose(2, 0, 1)
    
        label = label.get_fdata()
        label = label.transpose(2, 0, 1)

        # choose prepocessing based on the category.
        if ct_preprocessing_decision:
            image = preprocessing_ct(image)
        else:
            image = preprocessing_mr(image)
        
        # save the images and labels as pytorch files.
        save_pt(image, label, save_dir)

    
    
    
    # create a list of all of the images in the folders
    

    # start the conversion of training images and training labels.


    # save the images in the new file structure


    # convert the test images.
# the 

def prepare_conversion(cfg):
    """Collect the folders for the creation of the PyTorch files."""


    root_dir = cfg["data_storage"]["data_location"]
    pt_dir = cfg["data_storage"]["pt_location"]

    # create the root_dir path in the file system.
    Path(root_dir).mkdir(parents=False, exist_ok=True)
    Path(pt_dir).mkdir(parents=False, exist_ok=True)


    # select the folders to convert to pt
    brain_dir = os.path.join(root_dir, "Task01_BrainTumour")
    heart_dir = os.path.join(root_dir, "Task02_Heart")
    liver_dir = os.path.join(root_dir, "Task03_Liver")
    hippo_dir = os.path.join(root_dir, "Task04_Hippocampus")
    prostate_dir = os.path.join(root_dir, "Task05_Prostate")
    lung_dir = os.path.join(root_dir, "Task06_Lung")
    pancreas_dir = os.path.join(root_dir, "Task07_Pancreas")
    vessel_dir = os.path.join(root_dir, "Task08_HepaticVessel")
    spleen_dir = os.path.join(root_dir, "Task09_Spleen")
    colon_dir = os.path.join(root_dir, "Task10_Colon")

    # create new folders from names for the tasks
    folder_list = ["Task01_BrainTumor", "Task02_Heart", "Task03_Liver", "Task04_Hippocampus", "Task05_Prostate", "Task06_Lung", "Task07_Pancreas", "Task08_HepaticVessel", "Task09_Spleen", "Task10_Colon"]

    # create new folders from names in folder list in pt_dir
    for folder in folder_list:
        Path(os.path.join(pt_dir, folder)).mkdir(parents=False, exist_ok=True)
        # create a new folder for training and one for testing.
        Path(os.path.join(pt_dir, folder, "train")).mkdir(parents=False, exist_ok=True)
        Path(os.path.join(pt_dir, folder, "test")).mkdir(parents=False, exist_ok=True)








def train_test_split(root_dir, cfg):
    pass


def main():
    # define the data directory
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)


    # TODO create the folders for the data.

    root_dir = cfg["data_storage"]["data_location"]


    # start by downloading the data

    download(root_dir, cfg)

    # then extract the data

    # converd the data to pt files.


def download(root_dir, cfg):
    get_brain_aws = cfg["aws_links"]["brain"]
    get_heart_aws = cfg["aws_links"]["heart"]
    get_liver_aws = cfg["aws_links"]["liver"]
    get_hippo_aws = cfg["aws_links"]["hippo"]
    get_prostata_aws = cfg["aws_links"]["prostata"]
    get_lung_aws = cfg["aws_links"]["lung"]
    get_pancreas_aws = cfg["aws_links"]["pancreas"]
    get_vessel_aws = cfg["aws_links"]["vessel"]
    get_spleen_aws = cfg["aws_links"]["spleen"]
    get_colon_aws = cfg["aws_links"]["colon"]

    # Brain Tumor
    compressed_file = os.path.join(root_dir, "Task01_BrainTumour.tar")
    data_dir = os.path.join(root_dir, "Task01_BrainTumour")
    if not os.path.exists(data_dir):
        download_and_extract(get_brain_aws, compressed_file, root_dir)
    # Heart
    compressed_file = os.path.join(root_dir, "Task02_Heart.tar")
    data_dir = os.path.join(root_dir, "Task02_Heart")
    if not os.path.exists(data_dir):
        download_and_extract(get_heart_aws, compressed_file, root_dir)

    compressed_file = os.path.join(root_dir, "Task03_Liver.tar")
    data_dir = os.path.join(root_dir, "Task03_Liver")
    if not os.path.exists(data_dir):
        download_and_extract(get_liver_aws, compressed_file, root_dir)

    compressed_file = os.path.join(root_dir, "Task04_Hippocampus.tar")
    data_dir = os.path.join(root_dir, "Task04_Hippocampus")
    if not os.path.exists(data_dir):
        download_and_extract(get_hippo_aws, compressed_file, root_dir)

    compressed_file = os.path.join(root_dir, "Task05_Prostate.tar")
    data_dir = os.path.join(root_dir, "Task05_Prostate")
    if not os.path.exists(data_dir):
        download_and_extract(get_prostata_aws, compressed_file, root_dir)

    compressed_file = os.path.join(root_dir, "Task06_Lung.tar")
    data_dir = os.path.join(root_dir, "Task06_Lung")
    if not os.path.exists(data_dir):
        download_and_extract(get_lung_aws, compressed_file, root_dir)

    compressed_file = os.path.join(root_dir, "Task07_Pancreas.tar")
    data_dir = os.path.join(root_dir, "Task07_Pancreas")
    if not os.path.exists(data_dir):
        download_and_extract(get_pancreas_aws, compressed_file, root_dir)

    compressed_file = os.path.join(root_dir, "Task08_HepaticVessel.tar")
    data_dir = os.path.join(root_dir, "Task08_HepaticVessel")
    if not os.path.exists(data_dir):
        download_and_extract(get_vessel_aws, compressed_file, root_dir)

    compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
    data_dir = os.path.join(root_dir, "Task09_Spleen")
    if not os.path.exists(data_dir):
        download_and_extract(get_spleen_aws, compressed_file, root_dir)

    compressed_file = os.path.join(root_dir, "Task10_Colon.tar")
    data_dir = os.path.join(root_dir, "Task10_Colon")
    if not os.path.exists(data_dir):
        download_and_extract(get_colon_aws, compressed_file, root_dir)


if __name__ == "__main__":
    main()
