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

def doUncropping(inpDir: str):
    croppedDir = os.path.join(inpDir, 'pred_segmentations_cropped')
    uncroppedDir = os.path.join(inpDir, 'pred_segmentations')
    os.makedirs(uncroppedDir, exist_ok=True)
    for imgName in os.listdir(croppedDir):
        if not imgName.endswith('.nii.gz'):
            continue
        patientID = imgName.split('.')[0]
        patientInfoPath = os.path.join(r'E:\MAMA-MIA\patient_info_files', f'{patientID}.json')
        with open(patientInfoPath, 'r') as f:
            patientInfo = json.load(f)
        imgPath = os.path.join(croppedDir, imgName)
        ogImgPath = os.path.join(r'E:\MAMA-MIA\images', patientID.upper(), f'{patientID}_0000.nii.gz')
        getBoundingBox(ogImgPath, patientID, patientInfo)
        uncroppedImg = uncropToOriginalCoords(imgPath, patientID)
        sitk.WriteImage(uncroppedImg, os.path.join(uncroppedDir, f'{patientID}.nii.gz'))

if __name__ == "__main__":
    inpDir = "./outputs-64patch"
    # doUncropping(inpDir)
    generate_scores(inpDir)