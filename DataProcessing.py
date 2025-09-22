import os, glob
import torch
import zarr, math
import numpy as np
import SimpleITK as sitk
import pandas as pd
from functools import partial
from itertools import product
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from helpers import PATCH_SIZE, NUM_PATCHES, ACQ_TIME_THRESHOLD, MIN_NUM_PHASES
from numba import njit

@njit
def extractPatches(array, patch_indices, patch_size):
    n_patches = len(patch_indices)
    patches = np.empty((n_patches, patch_size, patch_size, patch_size), dtype=array.dtype)
    for i in range(n_patches):
        x, y, z = patch_indices[i]
        patches[i] = array[x:x+patch_size, y:y+patch_size, z:z+patch_size]
    return patches

def runHandles(handles):
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda h: h(), handles))

    # results = [h() for h in handles]

    targets, distMaps, phases, pcrs, patchIndices = zip(*results)
    return np.stack(targets), np.stack(distMaps), np.stack(phases), np.stack(pcrs), np.stack(patchIndices)

''' 
data {
    tr/val/ts: {                    ("training", "validation", "testing")
        num_phases: {
            patient_id: {           e.g. ("duke_002", "ispy2_981664")
                handle - partial that returns
                - segImg tensors
                    [NUM_PATCHES x PATCH_SIZE x PATCH_SIZE x PATCH_SIZE]
                - dmapImg tensors
                    [NUM_PATCHES x PATCH_SIZE x PATCH_SIZE x PATCH_SIZE]
                - phaseImg tensors
                    [num_phases x NUM_PATCHES x PATCH_SIZE x PATCH_SIZE x PATCH_SIZE]
                - pcr tensors
                    [singleton]
                - patchIndices
                    (list of 3-tuples)
            }
        }
    }
}
'''
# Initialize the data structure
def defaultPhaseDict():
    return defaultdict(dict)

def defaultSplitDict():
    return defaultdict(defaultPhaseDict)

def GetData(parentDir: str, patientDataPath: str, oversample: float = 0., test: bool = False):
    print("Collecting data...")

    # Load patient metadata once
    df = pd.read_excel(patientDataPath, "dataset_info")[["patient_id", "acquisition_times"]]
    np.random.seed(420)

    # Pickle-safe nested defaultdict
    data = defaultdict(defaultSplitDict)

    # Iterate through directories and organize the data
    for trts in os.listdir(parentDir):
        trtsDir = os.path.join(parentDir, trts)
        if not os.path.isdir(trtsDir):
            continue
        patientPaths = os.listdir(trtsDir)
        if test:
            patientPaths = [p for p in patientPaths if "nact" in p]
        
        # Iterate through patient images in the directory
        seenPatients = []
        for img in patientPaths:
            sp = img.split("_")
            patient_id = f"{sp[0]}_{sp[1]}"
            if patient_id in seenPatients:
                continue
            else:
                seenPatients.append(patient_id)
            
            # Determine the number of phases for the current patient
            acqTimes = df[df["patient_id"] == patient_id.upper()]["acquisition_times"].iloc[0]
            if not pd.isna(acqTimes):
                acqTimes = eval(acqTimes)
                nPhases = max(MIN_NUM_PHASES, sum([int(x <= ACQ_TIME_THRESHOLD) for x in acqTimes]))  # Calculate phases
            else:
                nPhases = MIN_NUM_PHASES

            # Get phase files for the patient
            phases = sorted(glob.glob(os.path.join(trtsDir, f"{patient_id}_0*.zarr")))[:nPhases]
            dmap = os.path.join(trtsDir, f"{patient_id}_dmap.zarr")
            seg = os.path.join(trtsDir, f"{patient_id}_seg.zarr")
            bboxPath = os.path.join(trtsDir, f"{patient_id}_bbox.txt")
            
            # Read bounding box
            with open(bboxPath, 'r') as f:
                bbox = eval(f.read().strip())

            # Load the image patches
            handle = loadImagePatches(phases, dmap, seg, PATCH_SIZE, NUM_PATCHES,
                                      oversample=oversample, fgBox=bbox, 
                                      loadWholeImage=(trts == "testing"))

            # Store data in the structure
            data[trts][nPhases][patient_id] = handle

    print("\tTraining and testing done, getting validation...")
    # Implement 10% split logic for validation
    data["validation"] = defaultdict(dict)

    # For each num_phases group
    for num_phases in list(data["training"].keys()):
        patient_dict = data["training"][num_phases]
        patient_ids = list(patient_dict.keys())

        for patient_id in patient_ids:
            if np.random.random() < 0.1:  # 10% chance
                # Move patient to validation
                data["validation"][num_phases][patient_id] = patient_dict[patient_id]
                del data["training"][num_phases][patient_id]

        # If a num_phases group ends up empty in training, remove it
        if not data["training"][num_phases]:
            del data["training"][num_phases]

    def countPatients(split_data: dict) -> int:
        return sum(len(patients) for patients in split_data.values())

    print(f"Got data: train = {countPatients(data['training'])}, "
          f"val = {countPatients(data['validation'])}, "
          f"test = {countPatients(data['testing'])}")

    return dict(data)  # Convert defaultdict to regular dict for return


# i feel very clever for this approach ;)
# Store partial handles in a data structure
# and then run them only when I need them (making sure to delete the arrays after to save memory)
# return segImg handles, dmapImg handles, [phaseImg handles for p in nPhases] 
def loadImagePatches(phasePaths: list[str], dmapPath: str, segPath: str, patchSize: int, 
                     numPatches: int, oversample: float, 
                     fgBox: tuple[list[int]], loadWholeImage: bool = False):
    assert all(path.endswith('.zarr') for path in [segPath] + [dmapPath] + phasePaths), "Only .zarr files are supported."

    segZarr     = zarr.open(segPath, mode='r')
    dmapZarr    = zarr.open(dmapPath, mode='r') if os.path.exists(dmapPath) else None
    phaseZarrs  = [zarr.open(p, mode='r') for p in phasePaths]

    return partial(getPatches, phaseZarrs, dmapZarr, segZarr, patchSize, numPatches, oversample, fgBox, loadWholeImage)

# need to select patch indices inside of function handle since we want it to be consistent for all images in a case,
# but it should pick different patches each iteration
def getPatches(phaseZarrs: list[zarr.Array], dmapZarr: zarr.Array, segZarr: zarr.Array, patchSize: int,
               numPatches: int, oversample: float, fgBox: tuple[list[int]], loadWholeImage: bool):
    shape = segZarr.shape
    patchIndices = getIndices(numPatches, patchSize, shape, 
                              oversample=oversample, fgBox=fgBox, 
                              loadWholeImage=loadWholeImage)

    segImgs      = extractPatches(segZarr[...].astype(np.int32), patchIndices, patchSize)

    # This one allowed to be None because don't need it for testing, and might not need it if no Boundary Loss
    if dmapZarr:
        dmapImgs = extractPatches(dmapZarr[...].astype(np.int32), patchIndices, patchSize)
    else:
        dmapImgs = None
    
    phaseVolumes = [z[...].astype(np.float32) / np.max(z) for z in phaseZarrs]
    phaseImgs    = np.stack([extractPatches(vol, patchIndices, patchSize) for vol in phaseVolumes])
    
    # TODO: Finish me!
    pcrValues    = None
    
    return segImgs, dmapImgs, phaseImgs, pcrValues, patchIndices

# randomly sample N patches
def getIndices(numPatches: int, patchSize: int, imgShape: list[int], oversample: float, fgBox: tuple[list[int]], loadWholeImage: bool = False):
    if loadWholeImage:
        pX, pY, pZ = getFullImageIndices(imgShape, patchSize, minOverlap=patchSize // 4)
        return list(product(pX, pY, pZ))
    
    maxXYZ = [c - patchSize for c in imgShape]
    minFG, maxFG = fgBox
    indices = []
    for _ in range(numPatches):
        if np.random.rand() < oversample:
            # sample the start index from foreground box, but make sure we don't go out of bounds
            r1 = range(max(minFG[0] - patchSize // 2, 0), min(maxFG[0] - patchSize // 2, maxXYZ[0]))
            r2 = range(max(minFG[1] - patchSize // 2, 0), min(maxFG[1] - patchSize // 2, maxXYZ[1]))
            r3 = range(max(minFG[2] - patchSize // 2, 0), min(maxFG[2] - patchSize // 2, maxXYZ[2]))
            start = (np.random.choice(r1),
                     np.random.choice(r2),
                     np.random.choice(r3))
        else:
            start = (np.random.choice(range(maxXYZ[0])),
                     np.random.choice(range(maxXYZ[1])),
                     np.random.choice(range(maxXYZ[2])))
        indices.append(start)
    return indices

def getFullImageIndices(imageShape, patchSize, minOverlap):
    def computeIdealStride(imageSize, patchSize, minOverlap):
        assert patchSize <= imageSize, "Patch must be smaller than or equal to image size"
        maxStride = patchSize - minOverlap
        numPatches = math.ceil((imageSize - patchSize) / maxStride) + 1
        stride = (imageSize - patchSize) // (numPatches - 1) + 1 if numPatches > 1 else 0
        return stride
    
    def computeAxisIndices(imageLen, patchLen, strideLen):
        indices = list(range(0, max(imageLen - patchLen + 1, 1), strideLen))
        if indices[-1] + patchLen < imageLen:
            indices.append(imageLen - patchLen)
        return indices
    
    x, y, z = imageShape
    xStride = computeIdealStride(x, patchSize, minOverlap)
    yStride = computeIdealStride(y, patchSize, minOverlap)
    zStride = computeIdealStride(z, patchSize, minOverlap)
    
    xIndices = computeAxisIndices(x, patchSize, xStride)
    yIndices = computeAxisIndices(y, patchSize, yStride)
    zIndices = computeAxisIndices(z, patchSize, zStride)

    return xIndices, yIndices, zIndices

def reconstructImageFromPatches(patches: list[torch.Tensor], patchCoords: list[torch.Tensor], patchSize: int):
    # Determine the target dimensions from the maximum coordinates in patchCoords
    max_coords = torch.max(torch.cat(patchCoords), dim=0).values
    target_dims = max_coords + patchSize
    target_dims = target_dims.tolist()

    # Initialize the reconstructed image and a weight map for averaging overlaps
    reconstructed_image: torch.Tensor = torch.zeros(target_dims, device=patches[0].device, dtype=float)
    weight_map: torch.Tensor = torch.zeros(target_dims, device=patches[0].device, dtype=float)

    for chunk, coords in zip(patches, patchCoords):
        for patch, coord in zip(chunk, coords):
            # Calculate the slice indices for the patch
            d_start, h_start, w_start = coord
            d_end, h_end, w_end = coord + patchSize

            # Add the patch to the reconstructed image and update the weight map
            reconstructed_image[d_start:d_end, h_start:h_end, w_start:w_end] += patch.float()
            weight_map[d_start:d_end, h_start:h_end, w_start:w_end] += 1.

    # Normalize the reconstructed image by the weight map to handle overlaps
    reconstructed_image /= torch.clamp(weight_map, min=1)

    return reconstructed_image.to(chunk.dtype)

def subpatchTensor(x: torch.Tensor, subpatchSize: int):
    assert len(x.shape) == 5, f"Expected shape (B, C, X, Y, Z). Got: {x.shape}"
    B, T, X, Y, Z = x.shape
    assert X % subpatchSize == 0 and Y % subpatchSize == 0 and Z % subpatchSize == 0, \
        "Input dimensions must be divisible by subpatchSize"

    # Reshape into subpatches
    x = x.unfold(2, subpatchSize, subpatchSize)  # Unfold X
    x = x.unfold(3, subpatchSize, subpatchSize)  # Unfold Y
    x = x.unfold(4, subpatchSize, subpatchSize)  # Unfold Z
    x = x.permute(0, 2, 3, 4, 1, 5, 6, 7)  # [B, X//subpatchSize, Y//subpatchSize, Z//subpatchSize, C, subpatchSize, subpatchSize, subpatchSize]
    x = x.reshape(B, -1, T, subpatchSize, subpatchSize, subpatchSize)  # [B, N, T, subpatchSize, subpatchSize, subpatchSize]
    x = x.unsqueeze(3)          # Add a singleton channel dimension for patch embedding: [B, N, T, 1, subpatchSize, subpatchSize, subpatchSize]
    x = x.permute(0, 2, 1, 3, 4, 5, 6)      # [B, T, N, 1, subpatchSize, subpatchSize, subpatchSize]

    numSubpatches = (X // subpatchSize) * (Y // subpatchSize) * (Z // subpatchSize)
    return x, numSubpatches

def GetBinaryArrayFromTrainingTensor(tensor: torch.Tensor):
    maxIndices = torch.argmax(tensor, dim=2, keepdim=True)

    binaryTensor = torch.zeros_like(tensor)  # output tensor with same dimensions
    binaryTensor.scatter_(1, maxIndices, 1)  # write 1's into all the indices from argmax

    return binaryTensor.int()

def ConvertBinaryNPArrayToImage(ndarr: np.ndarray):
    argmax = np.argmax(ndarr, axis=0)
    return sitk.Cast(sitk.GetImageFromArray(argmax), sitk.sitkInt32)

def ConvertSoftmaxTensorToBinaryImage(tensor: torch.Tensor) -> sitk.Image:
    arr = torch.argmax(tensor, dim=0).cpu().detach().numpy()
    return sitk.Cast(sitk.GetImageFromArray(arr), sitk.sitkInt32)

def Crop(image, nonzeroCoords: np.ndarray) -> sitk.Image:
    # Transpose the list of coordinates to get min/max for each axis
    xyzCoords = nonzeroCoords[:, ::-1]

    minIndices = np.min(xyzCoords, axis=0).astype(int).tolist()
    maxIndices = np.max(xyzCoords, axis=0).astype(int).tolist()

    size = [(maxIdx - minIdx + 1) for minIdx, maxIdx in zip(minIndices, maxIndices)]

    cropped = sitk.RegionOfInterest(image, size=size, index=minIndices)

    return cropped

if __name__ == "__main__":
    print("Starting program!")
    filePath = r"E:\MAMA-MIA\my_preprocessed_data\Dataset106_cropped_Xch_breast_no_norm\training\ispy2_738041_seg.zarr"
    outFolder = os.path.dirname(os.path.dirname(filePath))
    arr = zarr.load(filePath)
    # print(arr.dtype)
    if "seg" in filePath:
        arr = arr.astype(int)
    nii = sitk.GetImageFromArray(arr)
    sitk.WriteImage(nii, os.path.join(outFolder, "og.nii"))
    print("Wrote og.nii")

    patchSize = 32
    dev = torch.device("cuda")

    handles, indices = loadImagePatches(filePath, dev, patchSize=patchSize)

    allPatches  = []
    allCoords   = []
    for handle in handles:
        patches, coords = handle()
        # print(f"Patches shape: {patches.shape}")
        # print(f"Coords shape: {coords.shape}")
        allPatches.append(patches)
        allCoords.append(coords)

    print(f"there are {len(allPatches)} patches in the image")
    print(f"there are {len(allCoords)} coords")

    recon: torch.Tensor = reconstructImageFromPatches(allPatches, allCoords, patchSize, dev)
    if "seg" in filePath:
        recon = recon.int()

    sitk.WriteImage(sitk.GetImageFromArray(recon.cpu().detach().numpy()), os.path.join(outFolder, "recon.nii"))