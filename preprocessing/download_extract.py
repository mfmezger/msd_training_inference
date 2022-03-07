import nibabel as nib
import os
import random
import shutil
import torch
import yaml
from monai.apps import extractall
from parallel_sync import wget
from pathlib import Path
from monai.transforms import NormalizeIntensity


def save_pt(image, name, save_dir, mask=None):
    """Save the images and labels as pytorch files. If without mask it is considered to be the test image and will be stored without a mask."""

    # checking if train with mask or test without mask.
    if mask is None:
        image = torch.from_numpy(image)

        # path = save_dir + "/" + str(name) + ".pt"
        path = os.path.join(save_dir, str(name) + ".pt")

        torch.save({"vol": image, "id": name}, path)
    else:
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # path = save_dir + "/" + str(name) + ".pt"
        path = os.path.join(save_dir, str(name) + ".pt")

        torch.save({"vol": image, "mask": mask, "id": name}, path)


def convert_images(cfg, data_dir, save_dir):
    """Convert images to pytorch files."""

    # create lists of the images.
    label_list = os.listdir(os.path.join(data_dir, "labelsTr"))
    images_list = os.listdir(os.path.join(data_dir, "imagesTr"))
    image_list_test = os.listdir(os.path.join(data_dir, "imagesTs"))

    # remove the data that starts with . from the lists
    label_list = [x for x in label_list if not x.startswith(".")]
    images_list = [x for x in images_list if not x.startswith(".")]
    image_list_test = [x for x in image_list_test if not x.startswith(".")]

    # do the preprocessing based on the category.
    ct_list = ["Task03_Liver", "Task06_Lung", "Task07_Pancreas", "Task10_Colon", "Task08_HepaticVessel", "Task09_Spleen"]
    ct_preprocessing_decision = False
    if data_dir in ct_list:
        ct_preprocessing_decision = True

    normalize = NormalizeIntensity()

    # start the conversion of training images and training labels.
    # load the volumes from images and labels.
    for image, label in zip(images_list, label_list):
        # load the images and labels.
        image_path = os.path.join(data_dir, "imagesTr", image)
        label_path = os.path.join(data_dir, "labelsTr", label)

        # load the images and labels.
        image = nib.load(image_path)
        label = nib.load(label_path)

        image = image.get_fdata()
        label = label.get_fdata()
        # check for the datasets with more than 3 dimensions
        if len(image.shape) > 3:
            image = image.transpose(3, 2, 0, 1)
            label = label.transpose(2, 0, 1)

        else:
            # get image data and transpose to fit ZXY.
            image = image.transpose(2, 0, 1)

            label = label.transpose(2, 0, 1)

        # extract the name of the image.
        name = image_path.split("/")[-1].split(".")[0]

        # choose prepocessing based on the category.
        if ct_preprocessing_decision:
            image = normalize(image)    
        else:
            image = normalize(image)

        # save the images and labels as pytorch files.
        save_pt(image, name, save_dir + "/train", label)

    # save the images in the new file structure

    # convert the test images.
    for image in image_list_test:
        # load the images and labels.
        image_path = os.path.join(data_dir, "imagesTs", image)

        # extract the name of the image.
        name = image_path.split("/")[-1].split(".")[0]

        # load the images and labels.
        image = nib.load(image_path)
        image = image.get_fdata()

        if len(image.shape) > 2:
            image = image.transpose(3, 2, 0, 1)

        else:
            # get image data and transpose to fit ZXY.
            image = image.transpose(2, 0, 1)

        # choose prepocessing based on the category.
        if ct_preprocessing_decision:
            image = normalize(image)    
        else:
            image = normalize(image)

        # save the images and labels as pytorch files.
        save_pt(image, name, save_dir + "/test")


def prepare_conversion(cfg):
    """Collect the folders for the creation of the PyTorch files."""

    root_dir = cfg["data_storage"]["data_location"]
    pt_dir = cfg["data_storage"]["pt_location"]

    # create the root_dir path in the file system.
    Path(root_dir).mkdir(parents=False, exist_ok=True)
    Path(pt_dir).mkdir(parents=False, exist_ok=True)

    # retrive folder names.
    brain_dir, colon_dir, heart_dir, hippo_dir, liver_dir, lung_dir, pancreas_dir, prostate_dir, spleen_dir, vessel_dir = get_folders(root_dir)

    # create new folders from names for the tasks
    folder_list = ["Task01_BrainTumor", "Task02_Heart", "Task03_Liver", "Task04_Hippocampus", "Task05_Prostate", "Task06_Lung", "Task07_Pancreas", "Task08_HepaticVessel",
                   "Task09_Spleen", "Task10_Colon"]

    # create new folders from names in folder list in pt_dir
    for folder in folder_list:
        Path(os.path.join(pt_dir, folder)).mkdir(parents=False, exist_ok=True)
        # create a new folder for training and one for testing.
        Path(os.path.join(pt_dir, folder, "train")).mkdir(parents=False, exist_ok=True)
        Path(os.path.join(pt_dir, folder, "test")).mkdir(parents=False, exist_ok=True)

    # start the conversion to pt files.
    convert_images(cfg, brain_dir, os.path.join(pt_dir, "Task01_BrainTumor"))
    convert_images(cfg, heart_dir, os.path.join(pt_dir, "Task02_Heart"))
    convert_images(cfg, liver_dir, os.path.join(pt_dir, "Task03_Liver"))
    convert_images(cfg, hippo_dir, os.path.join(pt_dir, "Task04_Hippocampus"))
    convert_images(cfg, prostate_dir, os.path.join(pt_dir, "Task05_Prostate"))
    convert_images(cfg, lung_dir, os.path.join(pt_dir, "Task06_Lung"))
    convert_images(cfg, pancreas_dir, os.path.join(pt_dir, "Task07_Pancreas"))
    convert_images(cfg, vessel_dir, os.path.join(pt_dir, "Task08_HepaticVessel"))
    convert_images(cfg, spleen_dir, os.path.join(pt_dir, "Task09_Spleen"))
    convert_images(cfg, colon_dir, os.path.join(pt_dir, "Task10_Colon"))


def get_folders(root_dir):
    """Return the folder locations."""

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
    return brain_dir, colon_dir, heart_dir, hippo_dir, liver_dir, lung_dir, pancreas_dir, prostate_dir, spleen_dir, vessel_dir


def train_test_split(cfg):
    """Split the training data randomly in train and validation based on a   70%/30% split."""
    root_dir = cfg["data_storage"]["pt_location"]
    brain_dir, colon_dir, heart_dir, hippo_dir, liver_dir, lung_dir, pancreas_dir, prostate_dir, spleen_dir, vessel_dir = get_folders(root_dir)

    # get in every folder, randomly select 30% and move them to an extra validation folder. (move, not copy)
    for f in [heart_dir, brain_dir, colon_dir, hippo_dir, liver_dir, lung_dir, pancreas_dir, prostate_dir, spleen_dir, vessel_dir]:
        # create new val folder.
        Path(os.path.join(f, "validation")).mkdir(parents=False, exist_ok=True)
        # get the list of files in the folder.
        file_list = os.listdir(os.path.join(f, "train", ))
        # get the number of files in the folder.
        file_number = len(file_list)
        # get the number of files to move.
        move_number = int(file_number * 0.3)
        # get the list of files to move.
        move_list = random.sample(file_list, move_number)
        # move the files to the validation folder.
        for file in move_list:
            shutil.move(os.path.join(f, "train", file), os.path.join(f, "validation"))


def main():
    # define the data directory
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    root_dir = cfg["data_storage"]["data_location"]

    # start by downloading the data
    download(root_dir, cfg)

    # then extract the data
    prepare_conversion(cfg)

    # start the train val split.
    train_test_split(cfg)


def download(root_dir, cfg):
    """Download the data from AWS Open Data Repository."""

    # check if EU or NA server should be used.

    if cfg["data_storage"]["server"] == "EU":
        aws_links = "aws_links_EU"
    if cfg["data_storage"]["server"] == "NA":
        aws_links = "aws_links_NA"

    get_brain_aws = cfg[aws_links]["brain"]
    get_heart_aws = cfg[aws_links]["heart"]
    get_liver_aws = cfg[aws_links]["liver"]
    get_hippo_aws = cfg[aws_links]["hippo"]
    get_prostata_aws = cfg[aws_links]["prostata"]
    get_lung_aws = cfg[aws_links]["lung"]
    get_pancreas_aws = cfg[aws_links]["pancreas"]
    get_vessel_aws = cfg[aws_links]["vessel"]
    get_spleen_aws = cfg[aws_links]["spleen"]
    get_colon_aws = cfg[aws_links]["colon"]

    # Brain Tumor
    # compressed_file = os.path.join(root_dir, "Task01_BrainTumour.tar")
    # data_dir = os.path.join(root_dir, "Task01_BrainTumour")
    # if not os.path.exists(compressed_file):
    #     wget.download(get_brain_aws, compressed_file)
    #     extractall(get_brain_aws, compressed_file, data_dir)

    # Heart
    # FIXME: there is the Task02 folder twice when generating the subsystem.
    # compressed_file = os.path.join(root_dir, "Task02_Heart.tar")
    # data_dir = os.path.join(root_dir, "Task02_Heart")
    # if not os.path.exists(compressed_file):
    #     wget.download(root_dir, get_heart_aws)
    #     extractall(compressed_file, data_dir)

    # Liver
    compressed_file = os.path.join(root_dir, "Task03_Liver.tar")
    data_dir = os.path.join(root_dir, "Task03_Liver")
    if not os.path.exists(compressed_file):
        wget.download(root_dir, get_liver_aws)
        extractall(compressed_file, data_dir)

    # Hippocampus
    compressed_file = os.path.join(root_dir, "Task04_Hippocampus.tar")
    data_dir = os.path.join(root_dir, "Task04_Hippocampus")
    if not os.path.exists(compressed_file):
        wget.download(root_dir, get_hippo_aws)
        extractall(compressed_file, data_dir)

    # Prostata
    compressed_file = os.path.join(root_dir, "Task05_Prostate.tar")
    data_dir = os.path.join(root_dir, "Task05_Prostate")
    if not os.path.exists(compressed_file):
        wget.download(root_dir, get_prostata_aws)
        extractall(compressed_file, data_dir)

    # Lung
    compressed_file = os.path.join(root_dir, "Task06_Lung.tar")
    data_dir = os.path.join(root_dir, "Task06_Lung")
    if not os.path.exists(compressed_file):
        wget.download(root_dir, get_lung_aws)
        extractall(compressed_file, data_dir)

    # Pancreas
    compressed_file = os.path.join(root_dir, "Task07_Pancreas.tar")
    data_dir = os.path.join(root_dir, "Task07_Pancreas")
    if not os.path.exists(compressed_file):
        wget.download(root_dir, get_pancreas_aws)
        extractall(compressed_file, data_dir)

    # Hepatic Vessel
    compressed_file = os.path.join(root_dir, "Task08_HepaticVessel.tar")
    data_dir = os.path.join(root_dir, "Task08_HepaticVessel")
    if not os.path.exists(compressed_file):
        wget.download(root_dir, get_vessel_aws)
        extractall(compressed_file, data_dir)

    # Spleen
    compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
    data_dir = os.path.join(root_dir, "Task09_Spleen")
    if not os.path.exists(compressed_file):
        wget.download(root_dir, get_spleen_aws)
        extractall(compressed_file, data_dir)

    # Colon
    compressed_file = os.path.join(root_dir, "Task10_Colon.tar")
    data_dir = os.path.join(root_dir, "Task10_Colon")
    if not os.path.exists(compressed_file):
        wget.download(root_dir, get_colon_aws)
        extractall(compressed_file, data_dir)


if __name__ == "__main__":
    main()
