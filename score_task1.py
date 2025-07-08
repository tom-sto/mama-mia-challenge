import SimpleITK as sitk
import os
import json
from MAMAMIA.src.challenge.scoring_task1 import generate_scores

og_coords = {}

def getBoundingBox(imagePath: str, patient_id: str, patientInfo: dict):
    assert imagePath.endswith(".nii.gz"), f"Bad image path ending: {imagePath}"
    image = sitk.ReadImage(imagePath)
    
    boundingBox = patientInfo["primary_lesion"]["breast_coordinates"]
    
    zmin = boundingBox["x_min"]
    ymin = boundingBox["y_min"]
    xmin = boundingBox["z_min"]
    zmax = boundingBox["x_max"]
    ymax = boundingBox["y_max"]
    xmax = boundingBox["z_max"]
    
    # Store original coordinates and size for uncropping
    og_coords[patient_id] = {
        "original_size": image.GetSize(),
        "bounding_box": {
            "x_min": xmin, "x_max": xmax,
            "y_min": ymin, "y_max": ymax,
            "z_min": zmin, "z_max": zmax
        }
    }
    # print(og_coords)

def uncropToOriginalCoords(cropped_image_path: str, patient_id: str):
    cropped_image = sitk.ReadImage(cropped_image_path)
    # print(cropped_image.GetSize())
    original_size = og_coords[patient_id]["original_size"]
    bounding_box = og_coords[patient_id]["bounding_box"]

    # Create a blank image with the original size
    original_image = sitk.Image(original_size, cropped_image.GetPixelID())
    original_image.SetSpacing(cropped_image.GetSpacing())
    original_image.SetOrigin(cropped_image.GetOrigin())
    original_image.SetDirection(cropped_image.GetDirection())

    # Place the cropped image back into the original coordinate space
    x_min, x_max = bounding_box["x_min"], bounding_box["x_max"]
    y_min, y_max = bounding_box["y_min"], bounding_box["y_max"]
    z_min, z_max = bounding_box["z_min"], bounding_box["z_max"]

    original_array = sitk.GetArrayFromImage(original_image)
    cropped_array = sitk.GetArrayFromImage(cropped_image)

    original_array[z_min:z_max, y_min:y_max, x_min:x_max] = cropped_array

    uncropped_image = sitk.GetImageFromArray(original_array)
    uncropped_image.CopyInformation(original_image)

    return uncropped_image

def doUncropping(inpDir: str, forCorrupted: bool = False):
    croppedDir = os.path.join(inpDir, 'pred_segmentations_from_transformer_skips')
    uncroppedDir = os.path.join(inpDir, 'pred_segmentations')
    os.makedirs(uncroppedDir, exist_ok=True)
    for imgName in os.listdir(croppedDir):
        if not imgName.endswith('.nii.gz'):
            continue
        patientID = imgName.split('.')[0]      
        if forCorrupted:
            patientID = patientID[:-2]
        patientInfoPath = os.path.join(r'E:\MAMA-MIA\patient_info_files', f'{patientID}.json')
        with open(patientInfoPath, 'r') as f:
            patientInfo = json.load(f)
        imgPath = os.path.join(croppedDir, imgName)
        ogImgPath = os.path.join(r'E:\MAMA-MIA\images', patientID.upper(), f'{patientID}_0000.nii.gz')
        getBoundingBox(ogImgPath, patientID, patientInfo)
        uncroppedImg = uncropToOriginalCoords(imgPath, patientID)
        sitk.WriteImage(uncroppedImg, os.path.join(uncroppedDir, imgName))

def doCropping(inpDir: str):
    outputDir = inpDir + "_cropped"
    os.makedirs(outputDir, exist_ok=True)

    for imgName in os.listdir(inpDir):
        if not imgName.endswith('.nii.gz'):
            continue
        patientID = imgName.split('.')[0]
        patientInfoPath = os.path.join(r'E:\MAMA-MIA\patient_info_files', f'{patientID}.json')
        with open(patientInfoPath, 'r') as f:
            patientInfo = json.load(f)
        imgPath = os.path.join(inpDir, imgName)
        ogImgPath = os.path.join(r'E:\MAMA-MIA\images', patientID.upper(), f'{patientID}_0000.nii.gz')
        getBoundingBox(ogImgPath, patientID, patientInfo)
        bbox = og_coords[patientID]["bounding_box"]
        ogImg = sitk.ReadImage(imgPath)
        croppedImg: sitk.Image = sitk.RegionOfInterest(
            ogImg,
            size=[bbox["x_max"] - bbox["x_min"], bbox["y_max"] - bbox["y_min"], bbox["z_max"] - bbox["z_min"]],
            index=[bbox["x_min"], bbox["y_min"], bbox["z_min"]]
        )
        croppedImg.SetOrigin(croppedImg.GetOrigin())
        croppedImg.SetDirection(croppedImg.GetDirection())
        croppedImg.SetSpacing(croppedImg.GetSpacing())
        sitk.WriteImage(croppedImg, os.path.join(outputDir, imgName))
    return outputDir

def evaluateAcrossDatasets(inpDir: str):
    import pandas as pd

    # Load your CSV file into a DataFrame
    df = pd.read_csv(f'{inpDir}/results_task1.csv')

    # Filter and calculate average DSC for each group
    groups = ["DUKE", "ISPY1", "ISPY2", "NACT"]
    average_dsc = {}

    for group in groups:
        # Filter rows where patient_id contains the group and ignore NaN values in DSC
        group_df = df[df['patient_id'].str.contains(group, na=False)]
        average_dsc[group] = group_df['DSC'].mean()

    # Print the results
    for group, avg in average_dsc.items():
        print(f"Average DSC for {group}: {avg}")

def doScoring(inputDir, forCorrupted: bool = False):
    doUncropping(inputDir, forCorrupted)
    generate_scores(r"E:\MAMA-MIA", inputDir, forCorrupted)
    evaluateAcrossDatasets(inputDir)

if __name__ == "__main__":
    inpDir = r"C:\Users\stoughth\Documents\mama-mia\nnUNet_results\Dataset666_corrupted_preds\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_4_with_mri_64\results"
    # doUncropping(inpDir)
    generate_scores(r"E:\MAMA-MIA", inpDir, forCorrupted=True)