import os, shutil, json
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from os.path import join

PATIENT_INFO = r'D:\MAMA-MIA\patient_info_files'
prevDataset  = r'D:\MAMA-MIA\nnUNet_raw\Dataset104_cropped_3ch_breast'
newDataset   = r'D:\MAMA-MIA\nnUNet_raw\Dataset200_cropped_4ch_breast_and_seg'

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

def main():
    os.makedirs(newDataset, exist_ok=True)
    newTrDir = join(newDataset, 'imagesTr')
    newTsDir = join(newDataset, 'imageTs')
    newLbDir = join(newDataset, 'labelsTr')
    os.makedirs(newTrDir, exist_ok=True)
    os.makedirs(newTsDir, exist_ok=True)
    os.makedirs(newLbDir, exist_ok=True)

    trDir = join(prevDataset, 'imagesTr')
    tsDir = join(prevDataset, 'imagesTs')
    lbDir = r'D:\MAMA-MIA\segmentations\expert'

    import blosc2, numpy as np
    preprcDir = r"D:\MAMA-MIA\nnUNet_preprocessed_data\Dataset200_cropped_4ch_breast_and_seg\nnUNetPlans_3d_fullres"
    for fileName in os.listdir(preprcDir):
        if fileName.endswith("seg.b2nd") or fileName.endswith("pkl"):
            continue
        filePath = os.path.join(preprcDir, fileName)
        arr: blosc2.ndarray.NDArray = blosc2.open(filePath, mode='r')
        arr = blosc2.asarray(arr).squeeze()
        newArr = arr.copy()
        assert len(arr.shape) == 4 and arr.shape[0] == 4, f"Arr has shape {arr.shape}, expected (4, X, Y, Z)"
        dmap = arr[3]
        absMax = np.max(np.abs(dmap))
        dmap = dmap / absMax
        newArr[3] = dmap

        assert arr.shape == newArr.shape, f"Shapes don't match! {arr.shape} != {newArr.shape}"
        blosc2.asarray(np.ascontiguousarray(newArr), urlpath=filePath, mmap_mode='w+')

        print(f"Normalized {fileName}")

    # for imgName in os.listdir(trDir):
    #     # copy over first two channels
    #     ch = imgName.split('.')[0].split('_')[-1]
    #     patientID = imgName.split('.')[0][:-5]      # strip off channel
    #     if ch != '0000':
    #         continue
    #     imgPath = join(trDir, imgName)
    #     shutil.copy(imgPath, newTrDir)
    #     shutil.copy(join(trDir, f"{patientID}_0001.nii.gz"), newTrDir)

    #     # copy segmentation to 3rd channel
    #     segPath = join(lbDir, f"{patientID}.nii.gz")
    #     segImg = sitk.ReadImage(segPath)

    #     # crop seg to the right size!
    #     croppedSegImg = cropToBoundingBox(segImg, patientID)
    #     sitk.WriteImage(croppedSegImg, join(newTrDir, f"{patientID}_0002.nii.gz"))

    #     # copy distmap to 4th channel
    #     mask = sitk.GetArrayFromImage(croppedSegImg)
    #     edt = distance_transform_edt(mask)
    #     inv = distance_transform_edt(1 - mask.astype(int))
    #     sdm = inv - edt
    #     sdmImg = sitk.GetImageFromArray(sdm)
    #     sdmImg.CopyInformation(croppedSegImg)
    #     sitk.WriteImage(sdmImg, join(newTrDir, f"{patientID}_0003.nii.gz"))

    #     print(f"Finished {patientID}")

    # print("Done with Training!")
    # for imgName in os.listdir(tsDir):
    #     # copy over first two channels
    #     ch = imgName.split('.')[0].split('_')[-1]
    #     patientID = imgName.split('.')[0][:-5]      # strip off channel
    #     if ch != '0000':
    #         continue
    #     imgPath = join(tsDir, imgName)
    #     shutil.copy(imgPath, newTsDir)
    #     shutil.copy(join(tsDir, f"{patientID}_0001.nii.gz"), newTsDir)

    #     # copy segmentation to 3rd channel
    #     segPath = join(lbDir, f"{patientID}.nii.gz")
    #     segImg = sitk.ReadImage(segPath)

    #     # crop seg to the right size!
    #     croppedSegImg = sitk.Cast(cropToBoundingBox(segImg, patientID), sitk.sitkFloat32)
    #     sitk.WriteImage(croppedSegImg, join(newTsDir, f"{patientID}_0002.nii.gz"))

    #     # copy distmap to 4th channel
    #     mask = sitk.GetArrayFromImage(croppedSegImg)
    #     edt = distance_transform_edt(mask)
    #     inv = distance_transform_edt(1 - mask.astype(int))
    #     sdm = inv - edt         # THIS ONE HAS POSITIVE VALUES OUTSIDE, NEGATIVE VALUES INSIDE
    #     sdmImg = sitk.GetImageFromArray(sdm)
    #     sdmImg.CopyInformation(croppedSegImg)
    #     sitk.WriteImage(sdmImg, join(newTsDir, f"{patientID}_0003.nii.gz"))
    #     print(f"Finished {patientID}")

    # print("Done with Testing!")
    # # get pCR from spreadsheet and put it in the labels
    # print("Doing pcr labels")
    # import pandas as pd
    # import numpy as np
    # df = pd.read_excel(r"D:\MAMA-MIA\clinical_and_imaging_info.xlsx", sheet_name='dataset_info')
    # discard = []
    # for patientID in df['patient_id']:
    #     try:
    #         imgPath = join(newTrDir, f"{patientID}_0000.nii.gz")
    #         ogImg = sitk.ReadImage(imgPath)
    #     except:
    #         try:
    #             imgPath = join(newTsDir, f"{patientID}_0000.nii.gz")
    #             ogImg = sitk.ReadImage(imgPath)
    #         except:
    #             print("Cannot find patient images in new directories")
    #             return
    #     imgArr = sitk.GetArrayFromImage(ogImg)
    #     pcr = df.loc[df['patient_id'] == patientID, 'pcr']
    #     if pd.isna(pcr.item()):
    #         discard.append(patientID)
    #         print(f"No pcr label for {patientID}. Skipping")
    #         continue
    #     arr = np.ones_like(imgArr) * pcr.item()
    #     img = sitk.GetImageFromArray(arr)
    #     img.CopyInformation(ogImg)
    #     img = sitk.Cast(img, sitk.sitkUInt8)
    #     sitk.WriteImage(img, join(newLbDir, f"{patientID.lower()}.nii.gz"))

    # print("Done with PCR!")
    # print(f"Discard these images: {discard}")

if __name__ == '__main__':
    main()