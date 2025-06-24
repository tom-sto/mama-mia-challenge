import numpy as np
import os
# import sys
# sys.path.append('./MAMAMIA')
# from MAMAMIA.src.visualization import *
# from MAMAMIA.src.preprocessing import read_segmentation_from_patient_id

def WriteBinaryArrayToFile(arr, filepath, origin, spacing, direction):
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(arr)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    img.SetDirection(direction)
    sitk.WriteImage(img, filepath + ".nii.gz", useCompression=True, compressionLevel=-4)

def generateSegmentations():
    import torch
    from trainMyUNet import setupTrainer, inference
    datasetName = "Dataset104_cropped_3ch_breast"
    basepath = rf"{os.environ["nnUNet_preprocessed"]}\{datasetName}"
    pretrainedModelPath = None
    plansPath = rf"{basepath}\nnUNetPlans.json"
    datasetPath = rf"{basepath}\dataset.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    fold = 4
    trainer = setupTrainer(plansPath, "3d_fullres", fold, datasetPath, device, pretrainedModelPath, transformer=False)
    state_dict_path = rf"nnUNet_results\{datasetName}\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_{fold}\checkpoint_final.pth"
    testInputFolder = os.path.join(os.environ["nnUNet_raw"], datasetName, "imagesTs")

    inference(trainer, state_dict_path, testInputFolder, './testing')

    trainInputFolder = os.path.join(os.environ["nnUNet_raw"], datasetName, "imagesTr")
    valSegFolder = rf"nnUNet_results\{datasetName}\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_{fold}\validation"

    trainPaths = []
    valPatients = os.listdir(valSegFolder)
    testPatients = os.listdir("./testing")
    for patient in os.listdir(r"E:\MAMA-MIA\patient_info_files"):
        patientID = patient.split('.')[0]
        if patientID + ".nii.gz" in valPatients or patientID + ".nii.gz" in testPatients:
            continue
        baseImagePath = os.path.join(trainInputFolder, patientID)
        images = [baseImagePath + f"_000{i}.nii.gz" for i in range(3)]
        trainPaths.append(images)

    inference(trainer, state_dict_path, trainPaths, './training')

def generateKernel(shape: tuple[int], p: float = 0.5):
    kernel = np.random.rand(*shape)
    kernel = np.where(kernel < p, 1, 0).astype(np.uint8)
    return kernel

def corruptSegmentations(inputDir = './segDataRaw', outputDir=r'E:\MAMA-MIA\nnUNet_raw\Dataset666_corrupted_preds'):
    import SimpleITK as sitk
    from skimage.morphology import ball
    from scipy.ndimage import binary_dilation
    from edt import edt
    dirs = ["training", "validation", "testing"]
    dirs = []
    for d in dirs:
        os.makedirs(os.path.join(outputDir, d), exist_ok=True)
        dir = os.path.join(inputDir, d)
        for imgName in os.listdir(dir):
            imgPath = os.path.join(dir, imgName)
            patientID = imgName.split('.')[0]

            if os.path.exists(os.path.join(outputDir, d, patientID + f"_0_0000.nii.gz")):
                print(f"Skipping {patientID}...", end="\r")
                continue

            img = sitk.ReadImage(imgPath)
            origin, spacing, direction = img.GetOrigin(), img.GetSpacing(), img.GetDirection()
            arr = sitk.GetArrayFromImage(img)

            if np.all(arr == 0):
                ones = np.ones_like(arr).astype(np.float32) * -1.
                outArrs = [ones, ones, ones, ones]
                for i, array in enumerate(outArrs):
                    WriteBinaryArrayToFile(array, os.path.join(outputDir, d, patientID + f"_{i}_0000"), origin, spacing, direction)
                print()
                print(f"{patientID} had no foreground in this prediction, writing all -1 for augmentations.")
                print(f"Finished corrupting {imgPath}")
                continue

            if np.all(arr == 1):
                ones = np.ones_like(arr).astype(np.float32) * 1.
                outArrs = [ones, ones, ones, ones]
                for i, array in enumerate(outArrs):
                    WriteBinaryArrayToFile(array, os.path.join(outputDir, d, patientID + f"_{i}_0000"), origin, spacing, direction)
                print()
                print(f"{patientID} had no background in this prediction, writing all 1 for augmentations.")
                print(f"Finished corrupting {imgPath}")
                continue

            # if patientID == "duke_032":
            #     sitk.WriteImage(sitk.GetImageFromArray(ogArr), "img.mha")
            #     # ogog = read_segmentation_from_patient_id(r'E:\MAMA-MIA\segmentations\expert', patientID)
            #     # sitk.WriteImage(ogog, "og.mha")
            #     print(xmin, ymin, zmin, xmax, ymax, zmax)
            #     sitk.WriteImage(sitk.GetImageFromArray(cropped_arr), "cropped.mha")
            

            # img = cropped_image
            dilation_kernel = generateKernel(arr.shape, p=0.008)
            erosion_kernel = generateKernel(arr.shape, p=0.003)

            dilation_arr = arr.copy()
            erosion_arr = arr.copy()
            both_arr = arr.copy()

            for i in range(4):
                # 10% of the time, add a "new" tumor ball somewhere random
                if i == 0 and np.random.rand() < 0.1:
                    radius = np.random.choice(3) + 2    # pick a radius between 2 and 4
                    new_tumor = ball(radius)
                    new_tumor_mask: np.ndarray = np.zeros_like(dilation_arr)
                    larger_shape = new_tumor_mask.shape
                    smaller_shape = new_tumor.shape

                    # Randomly select a starting position
                    start_pos = [
                        np.random.randint(0, larger_shape[i] - smaller_shape[i] + 1)
                        for i in range(3)
                    ]

                    # Place the smaller array into the larger array
                    new_tumor_mask[
                        start_pos[0]:start_pos[0] + smaller_shape[0],
                        start_pos[1]:start_pos[1] + smaller_shape[1],
                        start_pos[2]:start_pos[2] + smaller_shape[2]
                    ] = new_tumor

                    dilation_arr = dilation_arr | new_tumor_mask
                    both_arr = both_arr | new_tumor_mask

                # dilate only
                nd_structuring_element = np.random.choice(2) + 1
                structuring_element = ball(nd_structuring_element, decomposition="sequence")
                masked_dilation_kernel = dilation_kernel & dilation_arr
                dilated_mask = binary_dilation(masked_dilation_kernel, structuring_element)    # DONT allow entirely new tumors to be created
                dilation_arr = dilated_mask | dilation_arr

                # erode only
                nd_structuring_element = np.random.choice(3) + 1
                structuring_element = ball(nd_structuring_element, decomposition="sequence")
                eroded_mask = binary_dilation(erosion_kernel, structuring_element)      # allow tumors to be eroded from the inside
                erosion_arr = ~eroded_mask & erosion_arr

                # dilate + erode
                both_arr = dilated_mask | both_arr
                both_arr = ~eroded_mask & both_arr

            outArrs = [arr, dilation_arr, erosion_arr, both_arr]

            for i, array in enumerate(outArrs):
                dist = edt(array)
                inv = edt(1 - array)
                edtArr = dist - inv     # THIS ONE HAS POSITIVE VALUES INSIDE FOREGROUND, NEGATIVE OUTSIDE
                absMax = np.max(np.abs(edtArr))
                normArr = edtArr / absMax
                WriteBinaryArrayToFile(normArr, os.path.join(outputDir, d, patientID + f"_{i}_0000"), origin, spacing, direction)

            print(f"Finished corrupting {imgPath}", end='\r')

    # for file in os.listdir(r"E:\MAMA-MIA\nnUNet_raw\Dataset666_corrupted_preds\imagesTr"):
    #     if not file.endswith("0000.nii.gz"):
    #         continue
    #     src = os.path.join(r"E:\MAMA-MIA\nnUNet_raw\Dataset666_corrupted_preds\imagesTr", file)
    #     img = sitk.ReadImage(src)
    #     arr = sitk.GetArrayFromImage(img)
    #     absMax = np.max(np.abs(arr))
    #     normArr = arr / absMax
    #     newImg = sitk.GetImageFromArray(normArr)
    #     newImg.CopyInformation(img)
    #     sitk.WriteImage(newImg, src)
    #     print(f"Normalized {file}", end='\r')

    # print("Done with training")
    # for file in os.listdir(r"E:\MAMA-MIA\nnUNet_raw\Dataset666_corrupted_preds\imagesTs"):
    #     if not file.endswith("0000.nii.gz"):
    #         continue
    #     src = os.path.join(r"E:\MAMA-MIA\nnUNet_raw\Dataset666_corrupted_preds\imagesTs", file)
    #     img = sitk.ReadImage(src)
    #     arr = sitk.GetArrayFromImage(img)
    #     absMax = np.max(np.abs(arr))
    #     normArr = arr / absMax
    #     newImg = sitk.GetImageFromArray(normArr)
    #     newImg.CopyInformation(img)
    #     sitk.WriteImage(newImg, src)
    #     print(f"Normalized {file}", end='\r')
    
    # print("Done with testing")

    # for file in os.listdir(r"E:\MAMA-MIA\nnUNet_raw\Dataset666_corrupted_preds\labelsTr"):
    #     src = os.path.join(r"E:\MAMA-MIA\nnUNet_raw\Dataset666_corrupted_preds\labelsTr", file)
    #     for i in range(4):
    #         dst = file.split('.')[0] + f'_{i}.nii.gz'
    #         dst = os.path.join(r"E:\MAMA-MIA\nnUNet_raw\Dataset666_corrupted_preds\labelsTr", dst)
    #         shutil.copy(src, dst)

    #     os.remove(src)

def getMRIs(mriDir: str = r"E:\MAMA-MIA\nnUNet_raw\Dataset104_cropped_3ch_breast", 
            corruptedDir: str = r"E:\MAMA-MIA\nnUNet_raw\Dataset666_corrupted_preds"):
    import shutil
    dirs = ["imagesTr", "imagesTs", "labelsTr"]
    # grab first MRI and put it as second channel

    for d in dirs:
        imgDir = os.path.join(mriDir, d)
        for imgName in os.listdir(imgDir):
            if not imgName.endswith("0001.nii.gz"):
                continue
            imgPath = os.path.join(imgDir, imgName)
            ogImgFirstPart = imgName[:-11]
            for imgSecondPart in [f"{i}_0002.nii.gz" for i in range(4)]:
                newImgName = ogImgFirstPart + imgSecondPart
                newImgPath = os.path.join(corruptedDir, d, newImgName)
                
                shutil.copy(imgPath, newImgPath)

            print(f"Finished copying {imgPath} to {newImgPath}", end='\r')

    print()
    print("Done!")

def main():
    # generateSegmentations()
    corruptSegmentations()
    getMRIs()
    

if __name__ == "__main__":
    main()