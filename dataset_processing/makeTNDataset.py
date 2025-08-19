import SimpleITK as sitk
import numpy as np
import pandas as pd
import zarr, json
import os, glob
from scipy.ndimage import distance_transform_edt as edt
from os.path import join
from time import time
from concurrent.futures import ProcessPoolExecutor

RAW_DIR = r"E:\MAMA-MIA\nnUNet_raw\Dataset106_cropped_Xch_breast_no_norm"
PP_DIR  = r"E:\MAMA-MIA\my_preprocessed_data\Dataset106_cropped_Xch_breast_no_norm"
TR_DIR = join(RAW_DIR, "imagesTr")
TS_DIR = join(RAW_DIR, "imagesTs")
LB_DIR = r"E:\MAMA-MIA\segmentations\expert"
PATIENT_INFO = r"E:\MAMA-MIA\patient_info_files"

def crop_to_bounding_box(image: sitk.Image, patientID: str):
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

def pad_to_patch_compatible_size(img: sitk.Image, patch_size: int = 32) -> sitk.Image:
    size = np.array(img.GetSize())
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()

    # Calculate padding for each dimension
    pad = (-(size % patch_size) + patch_size) % patch_size
    pad_before = pad // 2
    pad_after = pad - pad_before

    # Create new size with padding
    new_origin = tuple(origin[i] - pad_before[i] * spacing[i] for i in range(3))

    # Create padded image
    padded_img: sitk.Image = sitk.ConstantPad(img, pad_before.tolist(), pad_after.tolist(), constant=0)
    padded_img.SetOrigin(new_origin)
    padded_img.SetSpacing(spacing)
    padded_img.SetDirection(direction)

    return padded_img

def reorient_and_resample(img: sitk.Image,
                          target_orientation="RAS",
                          new_spacing=(1.0, 1.0, 1.0),
                          interpolator=sitk.sitkLinear):

    # Step 1: Fully reorient voxel data to target orientation
    oriented_img = sitk.DICOMOrient(img, target_orientation)

    # Step 2: Get original physical size
    size = oriented_img.GetSize()
    spacing = oriented_img.GetSpacing()

    physical_size = [spacing[i] * (size[i] - 1) for i in range(3)]

    # Step 3: Calculate new size based on new spacing
    new_size = [int(round(physical_size[i] / new_spacing[i]) + 1) for i in range(3)]

    # Step 4: Perform final resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(oriented_img.GetOrigin())
    resampler.SetOutputDirection(oriented_img.GetDirection())
    resampler.SetOutputPixelType(oriented_img.GetPixelID())
    resampler.SetTransform(sitk.Transform())  # identity

    resampled = resampler.Execute(oriented_img)
    return resampled

def do_training_case(patient_id, TR_DIR, PP_DIR, LB_DIR):
    print(f"Processing {patient_id}")
    outPathSeg = join(PP_DIR, "training", patient_id + "_seg.zarr")
    outPathDmap = join(PP_DIR, "training", patient_id + "_dmap.zarr")

    for file_path in glob.glob(os.path.join(TR_DIR, patient_id + "*.nii.gz")):
        outPath = join(PP_DIR, "training", file_path.split("\\")[-1][:-7])
        img = sitk.Cast(sitk.ReadImage(file_path), sitk.sitkFloat32)
        resImg = reorient_and_resample(img)
        paddedImg = pad_to_patch_compatible_size(resImg)

        arr = sitk.GetArrayFromImage(paddedImg)
        zarr.save(outPath + ".zarr", arr)
        # sitk.WriteImage(sitk.GetImageFromArray(arr), outPath + ".nii")
    print(f"\tFinished phases for {patient_id}")

    segImgPath = join(LB_DIR, patient_id + ".nii.gz")
    segImg = sitk.ReadImage(segImgPath)
    croppedSeg = crop_to_bounding_box(segImg, patient_id)
    resSeg = reorient_and_resample(croppedSeg, interpolator=sitk.sitkNearestNeighbor)
    paddedSeg = pad_to_patch_compatible_size(resSeg)

    seg = sitk.GetArrayFromImage(paddedSeg).astype(bool)
    zarr.save(outPathSeg, seg)
    print(f"\tFinished seg for {patient_id}")
    
    # get dmap for training set
    dist: np.ndarray = edt(seg)         # positive in foreground, 0 background
    inv: np.ndarray = edt(~seg)         # positive in background, 0 foreground
    dmap: np.ndarray = inv - dist       # negative in foreground, positive in background

    max_val = np.max(np.abs(dmap))
    if max_val != 0:
        dmap /= max_val
    
    zarr.save(outPathDmap, dmap.astype(np.float16))
    print(f"\tFinished dmap for {patient_id}")

    assert dmap.shape == seg.shape == arr.shape

def do_testing_case(patient_id, TS_DIR, PP_DIR, LB_DIR):
    print(f"Processing {patient_id}")
    outPathSeg = join(PP_DIR, "testing", patient_id + "_seg.zarr")

    for file_path in glob.glob(os.path.join(TS_DIR, patient_id + "*.nii.gz")):
        outPath = join(PP_DIR, "testing", file_path.split("\\")[-1][:-7])
        img = sitk.Cast(sitk.ReadImage(file_path), sitk.sitkFloat32)
        resImg = reorient_and_resample(img)
        paddedImg = pad_to_patch_compatible_size(resImg)

        arr = sitk.GetArrayFromImage(paddedImg)
        zarr.save(outPath + ".zarr", arr)
    print(f"\tFinished phases for {patient_id}")

    segImgPath = join(LB_DIR, patient_id + ".nii.gz")
    segImg = sitk.ReadImage(segImgPath)
    croppedSeg = crop_to_bounding_box(segImg, patient_id)
    resSeg = reorient_and_resample(croppedSeg, interpolator=sitk.sitkNearestNeighbor)
    paddedSeg = pad_to_patch_compatible_size(resSeg)

    seg = sitk.GetArrayFromImage(paddedSeg)
    zarr.save(outPathSeg, seg)
    print(f"\tFinished seg for {patient_id}")

def training_wrapper(patient_id):
    do_training_case(patient_id, TR_DIR, PP_DIR, LB_DIR)

def testing_wrapper(patient_id):
    do_testing_case(patient_id, TS_DIR, PP_DIR, LB_DIR)

def FormatSeconds(s):
    if s < 60:
        return f"{s:.4f} seconds"
    minutes = s // 60
    seconds = s % 60
    if s < 3600:
        return f"{minutes:.0f} minute{"s" if minutes > 1 else ""} and {seconds:.4f} seconds"
    hours = minutes // 60
    minutes %= 60
    if s < 86400:
        return f"{hours:.0f} hour{"s" if hours > 1 else ""}, {minutes:.0f} minute{"s" if minutes > 1 else ""}, and {seconds:.4f} seconds"
    # should never take this long, but you never know...
    days = hours // 24
    hours %= 24
    return f"{days:.0f} day{"s" if days > 1 else ""}, {hours:.0f} hour{"s" if hours > 1 else ""}, {minutes:.0f} minute{"s" if minutes > 1 else ""}, and {seconds:.4f} seconds"

def main():
    df = pd.read_csv(r"E:\MAMA-MIA\train_test_splits.csv")
    start = time()
    print("cleaning existing files...")
    if os.path.exists(join(PP_DIR, "training")):
        os.system(f'rmdir /S /Q "{join(PP_DIR, "training")}"')
    if os.path.exists(join(PP_DIR, "testing")):
        os.system(f'rmdir /S /Q "{join(PP_DIR, "testing")}"')
    print(f"took {FormatSeconds(time() - start)}.")
    
    os.makedirs(join(PP_DIR, "training"), exist_ok=True)
    os.makedirs(join(PP_DIR, "testing"), exist_ok=True)

    start = time()
    with ProcessPoolExecutor() as executor:
        list(executor.map(training_wrapper, df['train_split'].dropna().apply(str.lower).tolist()))
        list(executor.map(testing_wrapper, df['test_split'].dropna().apply(str.lower).tolist()))

    print(f"Took {FormatSeconds(time() - start)} to finish processing all data")

    return

if __name__ == "__main__":
    main()