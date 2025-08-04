import os, shutil, glob
import pandas as pd
import SimpleITK as sitk
import json
from MAMAMIA.nnUNet.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

RAW_PATH     = os.getenv("nnUNet_raw")
IMAGES_DIR   = r"E:\MAMA-MIA\images"
SEG_DIR      = r"E:\MAMA-MIA\segmentations\expert"
DATASET_ID   = "106"
DATASET_NAME = "cropped_Xch_breast_no_norm"
SPLIT_PATH   = r"E:\MAMA-MIA\train_test_splits.csv"
PATIENT_INFO = r"E:\MAMA-MIA\patient_info_files"

def cropToBoundingBox(imagePath: str, patientID: str):
    assert imagePath.endswith(".nii.gz"), f"Bad image path ending: {imagePath}"
    image = sitk.ReadImage(imagePath)
    
    patientInfoPath = os.path.join(PATIENT_INFO, f"{patientID.lower()}.json")
    with open(patientInfoPath) as f:
        patientInfo = json.load(f)
        boundingBox = patientInfo["primary_lesion"]["breast_coordinates"]
    
    zmin = boundingBox["x_min"]
    ymin = boundingBox["y_min"]
    xmin = boundingBox["z_min"]
    zmax = boundingBox["x_max"]
    ymax = boundingBox["y_max"]
    xmax = boundingBox["z_max"]
    
    # Crop the image using the bounding box
    cropped_image: sitk.Image = sitk.RegionOfInterest(
        image, 
        size=[xmax - xmin, ymax - ymin, zmax - zmin], 
        index=[xmin, ymin, zmin]
    )
    
    return cropped_image


def main():
    assert RAW_PATH is not None, "Set the environment variable first!"
    assert os.path.exists(RAW_PATH), f"{RAW_PATH} not found"
    assert os.path.exists(IMAGES_DIR), f"{IMAGES_DIR} not found"
    assert os.path.exists(SEG_DIR), f"{SEG_DIR} not found"
    dataset_path = os.path.join(RAW_PATH, f"Dataset{DATASET_ID}_{DATASET_NAME}")
    os.makedirs(dataset_path, exist_ok=True)

    df = pd.read_csv(SPLIT_PATH)

    train_dir   = os.path.join(dataset_path, "imagesTr")
    test_dir    = os.path.join(dataset_path, "imagesTs")
    label_dir   = os.path.join(dataset_path, "labelsTr")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for patientID in os.listdir(IMAGES_DIR):
        out_dir = train_dir if patientID in list(df["train_split"]) else test_dir
        for file_path in glob.glob(os.path.join(IMAGES_DIR, patientID, "*.nii.gz")):
            assert os.path.exists(file_path), f"{file_path} doesn't exist"
            croppedImage = cropToBoundingBox(file_path, patientID)
            sitk.WriteImage(croppedImage, os.path.join(out_dir, file_path.split("\\")[-1]))
    
    print("Done with Training/Testing data!")
    
    for patientID in df["train_split"].dropna():
        in_path = os.path.join(SEG_DIR, f"{str(patientID).lower()}.nii.gz")
        assert os.path.exists(in_path), f"{in_path} doesn't exist"
        croppedSegImage = cropToBoundingBox(in_path, patientID)
        sitk.WriteImage(croppedSegImage, os.path.join(label_dir, in_path.split("\\")[-1]))

    print("Done with segmentation data!")

    generate_dataset_json(
        output_folder=dataset_path,
        channel_names={0: "Pre-Contrast", 
                       1: "Post-Contrast 1", 
                       2: "Post-Contrast 2",
                       3: "Post-Contrast 3", 
                       4: "Post-Contrast 4", 
                       5: "Post-Contrast 5"
                       },
        labels={"background": 0, "tumor": 1},
        num_training_cases=1200,
        file_ending=".nii.gz",
        dataset_name=f"Dataset{DATASET_ID}_{DATASET_NAME}",
        converted_by="Tom"
    )

    print("Finished")

if __name__ == "__main__":
    main()