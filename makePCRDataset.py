import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import json
import sys
import concurrent.futures
sys.path.append('MAMAMIA')
from MAMAMIA.nnUNet.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from MAMAMIA.src.visualization import get_segmentation_bounding_box
from MAMAMIA.src.preprocessing import read_mri_phase_from_patient_id, read_segmentation_from_patient_id, zscore_normalization_sitk, resample_sitk

RAW_PATH        = os.getenv("nnUNet_raw")
IMAGES_DIR      = r"E:\MAMA-MIA\images"
SEG_DIR         = r"E:\MAMA-MIA\segmentations\expert"
DATASET_ID      = "200"
DATASET_NAME    = "global_local_4ch_breast"
SPLIT_PATH      = r"E:\MAMA-MIA\train_test_splits.csv"
PATIENT_INFO    = r"E:\MAMA-MIA\patient_info_files"
CLINICAL_DATA   = r"E:\MAMA-MIA\clinical_and_imaging_info.xlsx"
PATCH_SIZE      = (96, 96, 96)

def cropToBoundingBox(image: sitk.Image, patientID: str):
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

def ZScoreNorm(images: list[sitk.Image]):
    means = []
    stds = []
    for img in images:
        LabelFilter = sitk.StatisticsImageFilter()
        LabelFilter.Execute(img)
        means.append(LabelFilter.GetMean())
        stds.append(LabelFilter.GetSigma())

    # pooled statistics, these images should have exactly the same number of points (voxels), which makes it easier to calculate
    pooled_mean = np.mean(means)
    pooled_std  = np.sqrt(np.sum(np.square(stds)) / len(stds))

    return [zscore_normalization_sitk(img, pooled_mean, pooled_std) for img in images]

def expandTumorBoxToPatchSize(bbox: list[int], croppedImage: sitk.Image) -> sitk.Image:
    [x_min, y_min, z_min, x_max, y_max, z_max] = bbox
    # print(f"Tumor bbox is {bbox}")
    fullSize = croppedImage.GetSize()

    # Calculate the center of the original bounding box
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    center_z = (z_min + z_max) // 2

    # Calculate new bounding box dimensions
    patch_x_min = max(0, center_x - PATCH_SIZE[0] // 2)
    patch_y_min = max(0, center_y - PATCH_SIZE[1] // 2)
    patch_z_min = max(0, center_z - PATCH_SIZE[2] // 2)
    # print(f"center of the tumor is {(center_x, center_y, center_z)}")

    patch_x_max = min(fullSize[0], patch_x_min + PATCH_SIZE[0])
    patch_y_max = min(fullSize[1], patch_y_min + PATCH_SIZE[1])
    patch_z_max = min(fullSize[2], patch_z_min + PATCH_SIZE[2])

    # Adjust min values if max values are clipped
    patch_x_min = max(0, patch_x_max - PATCH_SIZE[0] if patch_x_max - patch_x_min < PATCH_SIZE[0] else patch_x_min)
    patch_y_min = max(0, patch_y_max - PATCH_SIZE[1] if patch_y_max - patch_y_min < PATCH_SIZE[1] else patch_y_min)
    patch_z_min = max(0, patch_z_max - PATCH_SIZE[2] if patch_z_max - patch_z_min < PATCH_SIZE[2] else patch_z_min)

    # Extract the patch
    # print(f"Image is size: {croppedImage.GetSize()}")
    # print(f"cropping to {patch_x_min}:{patch_x_max}, {patch_y_min}:{patch_y_max}, {patch_z_min}:{patch_z_max},")
    patchedImage = sitk.RegionOfInterest(
        croppedImage,
        size=[patch_x_max - patch_x_min, patch_y_max - patch_y_min, patch_z_max - patch_z_min],
        index=[patch_x_min, patch_y_min, patch_z_min]
    )

    # Pad with zeros if the image is smaller than PATCH_SIZE
    padLowerBound = [
        max(0, (PATCH_SIZE[0] - (patch_x_max - patch_x_min)) // 2),
        max(0, (PATCH_SIZE[1] - (patch_y_max - patch_y_min)) // 2),
        max(0, (PATCH_SIZE[2] - (patch_z_max - patch_z_min)) // 2)
    ]
    padUpperBound = [
        max(0, PATCH_SIZE[0] - (patch_x_max - patch_x_min) - padLowerBound[0]),
        max(0, PATCH_SIZE[1] - (patch_y_max - patch_y_min) - padLowerBound[1]),
        max(0, PATCH_SIZE[2] - (patch_z_max - patch_z_min) - padLowerBound[2])
    ]

    paddedImage: sitk.Image = sitk.ConstantPad(
        patchedImage,
        padLowerBound=padLowerBound,
        padUpperBound=padUpperBound,
        constant=0
    )

    return paddedImage

def process_patient(patientID: str, clinical_df: pd.DataFrame, df: pd.DataFrame, train_dir: str, test_dir: str, label_dir: str):
    pcr_label = clinical_df.loc[clinical_df['patient_id'] == patientID, 'pcr'].values[0]
    if np.isnan(pcr_label):
        print(f"No label for {patientID}, excluding from PCR training")
        return None

    out_dir = train_dir if patientID in list(df["train_split"]) else test_dir
    preImg = read_mri_phase_from_patient_id(IMAGES_DIR, patientID, phase=0)
    postImg = read_mri_phase_from_patient_id(IMAGES_DIR, patientID, phase=1)

    croppedPreImage = cropToBoundingBox(preImg, patientID)
    croppedPostImage = cropToBoundingBox(postImg, patientID)

    # do z-score normalization
    normPreImg, normPostImg = ZScoreNorm([croppedPreImage, croppedPostImage])

    # find tumor segmentation bounding box
    tumorImg = read_segmentation_from_patient_id(SEG_DIR, patientID)
    croppedTumorImg = cropToBoundingBox(tumorImg, patientID)
    bbox = get_segmentation_bounding_box(croppedTumorImg)
    prePatch = expandTumorBoxToPatchSize(bbox, normPreImg)
    postPatch = expandTumorBoxToPatchSize(bbox, normPostImg)

    # resample cropped image to patch size
    resPreImg: sitk.Image = resample_sitk(normPreImg, new_size=PATCH_SIZE, interpolator=sitk.sitkLinear)
    resPostImg: sitk.Image = resample_sitk(normPostImg, new_size=PATCH_SIZE, interpolator=sitk.sitkLinear)

    prePatch.SetDirection(resPreImg.GetDirection())
    prePatch.SetOrigin(resPreImg.GetOrigin())
    postPatch.SetDirection(resPostImg.GetDirection())
    postPatch.SetOrigin(resPostImg.GetOrigin())

    # set all spacings to 1 since we are not going to do any resampling for preprocessing!
    # and nnUNet complains if we have different spacings in the same patient
    resPreImg.SetSpacing((1, 1, 1))
    resPostImg.SetSpacing((1, 1, 1))
    prePatch.SetSpacing((1, 1, 1))
    postPatch.SetSpacing((1, 1, 1))

    sitk.WriteImage(resPreImg, os.path.join(out_dir, f"{patientID.lower()}_0000.nii.gz"))
    sitk.WriteImage(resPostImg, os.path.join(out_dir, f"{patientID.lower()}_0001.nii.gz"))
    sitk.WriteImage(prePatch, os.path.join(out_dir, f"{patientID.lower()}_0002.nii.gz"))
    sitk.WriteImage(postPatch, os.path.join(out_dir, f"{patientID.lower()}_0003.nii.gz"))

    pcrArr = np.ones(shape=PATCH_SIZE, dtype=int) * int(pcr_label)
    pcrImg: sitk.Image = sitk.Cast(sitk.GetImageFromArray(pcrArr), sitk.sitkInt16)
    pcrImg.SetDirection(resPreImg.GetDirection())
    pcrImg.SetOrigin(resPreImg.GetOrigin())
    pcrImg.SetSpacing(resPreImg.GetSpacing())

    sitk.WriteImage(pcrImg, os.path.join(label_dir, f"{patientID.lower()}.nii.gz"))

    print(f"Finished processing {patientID} in {out_dir}")
    return patientID

def main():
    assert RAW_PATH is not None, "Set the environment variable first!"
    assert os.path.exists(RAW_PATH), f"{RAW_PATH} not found"
    assert os.path.exists(IMAGES_DIR), f"{IMAGES_DIR} not found"
    assert os.path.exists(SEG_DIR), f"{SEG_DIR} not found"
    dataset_path = os.path.join(RAW_PATH, f"Dataset{DATASET_ID}_{DATASET_NAME}")
    os.makedirs(dataset_path, exist_ok=True)

    df = pd.read_csv(SPLIT_PATH)
    clinical_df = pd.read_excel(CLINICAL_DATA, 'dataset_info')

    train_dir   = os.path.join(dataset_path, "imagesTr")
    test_dir    = os.path.join(dataset_path, "imagesTs")
    label_dir   = os.path.join(dataset_path, "labelsTr")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    patientIDs = os.listdir(IMAGES_DIR)
    nCases = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_patient, patientID, clinical_df, df, train_dir, test_dir, label_dir)
            for patientID in patientIDs
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None and result in list(df["train_split"]):
                nCases += 1

    print("Done with Training/Testing data!")

    generate_dataset_json(
        output_folder=dataset_path,
        channel_names={0: "Pre-Contrast Downsampled", 
                       1: "Post-Contrast 1 Downsampled", 
                       2: "Pre-Contrast Tumor Patch",
                       3: "Post-Contrast Tumor Patch"
                       },
        labels={"background": 0, "tumor": 1},
        num_training_cases=nCases,
        file_ending=".nii.gz",
        dataset_name=f"Dataset{DATASET_ID}_{DATASET_NAME}",
        converted_by="Tom"
    )

    print("Finished")

if __name__ == "__main__":
    main()