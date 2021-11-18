import os

import yaml
from monai.apps import download_and_extract


def convert_to_pt(cfg):

    root_dir = cfg["data_storage"]["data_location"]
    pt_dir = cfg["data_storage"]["pt_location"]
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
        os.makedirs(os.path.join(pt_dir, folder))
    
    # load the volumes from images

    # create a list of all of the images in the folders
    brain_images_list = os.listdir(os.path.join( brain_dir, "imagesTr"))
    brain_label_list = os.listdir(os.path.join( brain_dir, "labelsTr"))
    brain_image_list_test = os.listdir(os.path.join( brain_dir, "imagesTs"))

    # start the conversion of training images and training labels.
    

    # save the images in the new file structure


    # convert the test images.





def train_test_split(root_dir, cfg):
    pass


def main():
    # define the data directory
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

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
