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
from helpers import PATCH_SIZE, NUM_PATCHES, ACQ_TIME_THRESHOLD, MIN_NUM_PHASES, DTYPE_DMAP, DTYPE_SEG, DTYPE_PHASE
from Augmenter import GetNoTransforms, GetTrainingTransforms, DoTransforms

@torch.jit.script
def extractPatches(array: torch.Tensor, patch_indices: list[list[int]], patch_size: int):
    n_patches = len(patch_indices)
    patches = torch.empty((n_patches, patch_size, patch_size, patch_size), dtype=array.dtype)
    for i in range(n_patches):
        idx: list[int] = patch_indices[i]
        x, y, z = idx
        patches[i] = array[x:x+patch_size, y:y+patch_size, z:z+patch_size]
    return patches

def runHandles(handles):
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda h: h(), handles))

    targets, distMaps, phases, pcrs, patchIndices = zip(*results)
    return torch.stack(targets), torch.stack(distMaps), torch.stack(phases), torch.stack(pcrs), torch.stack(patchIndices)

''' 
data {
    tr/val/ts: {                    ("training", "validation", "testing")
        num_phases: {
            patient_id:            e.g. ("duke_002", "ispy2_981664")
                handle (or [handles]) - partial that returns
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
'''
# Initialize the data structure
def defaultPhaseDict():
    return defaultdict(dict)

def defaultSplitDict():
    return defaultdict(defaultPhaseDict)

def GetData(parentDir: str, patientDataPath: str, test: bool = False, oversample: float = 0.,
            oversampleRadius: float = 0.25):
    print("Collecting data...")

    # Load patient metadata once
    df = pd.read_excel(patientDataPath, "dataset_info")[["patient_id", "acquisition_times"]]
    np.random.seed(42)

    trainingCompose = GetTrainingTransforms()
    valTestCompose = GetNoTransforms()

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

            segZarr     = zarr.open(seg, mode='r')
            dmapZarr    = zarr.open(dmap, mode='r') if os.path.exists(dmap) else None
            phaseZarrs  = [zarr.open(p, mode='r') for p in phases]
            
            bboxPath = os.path.join(trtsDir, f"{patient_id}_bbox.txt")
            # Read bounding box
            with open(bboxPath, 'r') as f:
                bbox = eval(f.read().strip())

            # Load the image patches - I feel very clever for this approach ;)
            # Store partial function handles in a data structure
            # and then run them only when I need them (making sure to delete the arrays after to save memory)
            # return segImg handles, dmapImg handles, [phaseImg handles for p in nPhases] 
            if trts == "training" and np.random.random() < 0.9:
                handle1 = partial(GetPatches, phaseZarrs, dmapZarr, segZarr, PATCH_SIZE, NUM_PATCHES, oversample, bbox, False, oversampleRadius, trainingCompose)
                handle2 = partial(GetPatches, phaseZarrs, dmapZarr, segZarr, PATCH_SIZE, NUM_PATCHES, oversample, bbox, False, oversampleRadius, trainingCompose)
                handle3 = partial(GetPatches, phaseZarrs, dmapZarr, segZarr, PATCH_SIZE, NUM_PATCHES, -1, bbox, False, oversampleRadius, trainingCompose)
                handle4 = partial(GetPatches, phaseZarrs, dmapZarr, segZarr, PATCH_SIZE, NUM_PATCHES, -1, bbox, False, oversampleRadius, trainingCompose)
                
                handles = [handle1, handle2, handle3, handle4]

                # Store data in the structure
                data[trts][nPhases][patient_id] = handles
            elif trts == "training":
                data["validation"][nPhases][patient_id] = partial(GetPatches, phaseZarrs, dmapZarr, segZarr, PATCH_SIZE, NUM_PATCHES, 
                                                                  0, bbox, True, oversampleRadius, valTestCompose)
            else:
                data[trts][nPhases][patient_id] = partial(GetPatches, phaseZarrs, dmapZarr, segZarr, PATCH_SIZE, NUM_PATCHES, 
                                                          0, bbox, True, oversampleRadius, valTestCompose)

    def countPatients(split_data: dict) -> int:
        return sum(len(patients) for patients in split_data.values())

    print(f"Got data: train = {countPatients(data['training'])}, "
          f"val = {countPatients(data['validation'])}, "
          f"test = {countPatients(data['testing'])}")

    return dict(data)  # Convert defaultdict to regular dict for return

# need to select patch indices inside of function handle since we want it to be consistent for all images in a case,
# but it should pick different patches each iteration
def GetPatches(phaseZarrs: list[zarr.Array], dmapZarr: zarr.Array, segZarr: zarr.Array, patchSize: int,
               numPatches: int, oversample: float, fgBox: tuple[list[int]], loadWholeImage: bool,
               oversampleRadius: float, augmentationCompose):
    shape = segZarr.shape
    patchIndices = getIndices(numPatches, patchSize, shape, 
                              oversample=oversample, fgBox=fgBox, 
                              loadWholeImage=loadWholeImage,
                              oversampleRadius=oversampleRadius)

    segArr  = torch.from_numpy(segZarr[...]).to(DTYPE_SEG)
    dmapArr = torch.from_numpy(dmapZarr[...]).to(DTYPE_DMAP) if dmapZarr is not None else None
    phaseArrs = [torch.from_numpy(vol[...].astype(np.float32) / np.max(vol)).to(DTYPE_PHASE) for vol in phaseZarrs]

    phaseArrs, segArr, dmapArr = DoTransforms(phaseArrs, segArr, dmapArr, augmentationCompose)

    segImgs = extractPatches(segArr, patchIndices, patchSize)

    # This one allowed to be None because don't need it for testing, and might not need it if no Boundary Loss
    if dmapArr is not None:
        dmapImgs = extractPatches(dmapArr, patchIndices, patchSize)
    else:
        dmapImgs = None
    
    phaseImgs = torch.stack([extractPatches(vol, patchIndices, patchSize) for vol in phaseArrs])
    
    # TODO: Finish me!
    pcrValues = torch.tensor([0])
    
    return segImgs, dmapImgs, phaseImgs, pcrValues, torch.tensor(patchIndices).int()

# randomly sample N patches
def getIndices(numPatches: int, patchSize: int, imgShape: list[int], oversample: float, fgBox: tuple[list[int]], loadWholeImage: bool,
               shuffleFullImageIndices: bool = True, oversampleRadius: float = 0.25):
    if loadWholeImage:
        pX, pY, pZ = getFullImageIndices(imgShape, patchSize, minOverlap=patchSize // 4)
        indices = list(product(pX, pY, pZ))
        indices = [list(c) for c in indices]
        if shuffleFullImageIndices:
            np.random.shuffle(indices)
        return indices
    
    maxXYZ = [c - patchSize for c in imgShape]
    minFG, maxFG = fgBox
    indices = []
    oversampleThreshold = np.clip(np.random.normal(oversample, oversampleRadius), 0, 1)
    for _ in range(numPatches):
        # make sure that our patches are sampled so that they all DO NOT contain tumor foreground
        if oversample == -1:
            # Build valid ranges in each dimension, excluding the foreground region
            validX = list(range(0, max(minFG[0] - patchSize + 1, 1))) + \
                     list(range(min(maxFG[0], maxXYZ[0] - 1), maxXYZ[0]))
            validY = list(range(0, max(minFG[1] - patchSize + 1, 1))) + \
                     list(range(min(maxFG[1], maxXYZ[1] - 1), maxXYZ[1]))
            validZ = list(range(0, max(minFG[2] - patchSize + 1, 1))) + \
                     list(range(min(maxFG[2], maxXYZ[2] - 1), maxXYZ[2]))

            # If no valid range in any dimension, fall back to full random sampling
            if not validX or not validY or not validZ:
                start = [np.random.choice(range(maxXYZ[0])),
                         np.random.choice(range(maxXYZ[1])),
                         np.random.choice(range(maxXYZ[2]))]
            else:
                start = [np.random.choice(validX),
                         np.random.choice(validY),
                         np.random.choice(validZ)]
        elif np.random.rand() < oversampleThreshold:
            # sample the start index from foreground box, but make sure we don't go out of bounds
            r1 = range(max(minFG[0] - patchSize // 2, 0), min(maxFG[0] - patchSize // 2, maxXYZ[0]))
            r2 = range(max(minFG[1] - patchSize // 2, 0), min(maxFG[1] - patchSize // 2, maxXYZ[1]))
            r3 = range(max(minFG[2] - patchSize // 2, 0), min(maxFG[2] - patchSize // 2, maxXYZ[2]))
            start = [np.random.choice(r1),
                     np.random.choice(r2),
                     np.random.choice(r3)]
        else:
            start = [np.random.choice(range(maxXYZ[0])),
                     np.random.choice(range(maxXYZ[1])),
                     np.random.choice(range(maxXYZ[2]))]
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

def reconstructImageFromPatches(patches: torch.Tensor, patchCoords: torch.Tensor, patchSize: int):
    # Determine the target dimensions from the maximum coordinates in patchCoords
    max_coords = torch.max(patchCoords, dim=0).values
    target_dims = max_coords + patchSize
    target_dims = target_dims.tolist()

    # Initialize the reconstructed image and a weight map for averaging overlaps
    reconstructed_image: torch.Tensor = torch.zeros(target_dims, device=patches[0].device, dtype=float)
    weight_map: torch.Tensor = torch.zeros(target_dims, device=patches[0].device, dtype=float)

    for patch, coord in zip(patches, patchCoords):
        # Calculate the slice indices for the patch
        d_start, h_start, w_start = coord
        d_end, h_end, w_end = coord + patchSize

        # Add the patch to the reconstructed image and update the weight map
        reconstructed_image[d_start:d_end, h_start:h_end, w_start:w_end] += patch.float()
        weight_map[d_start:d_end, h_start:h_end, w_start:w_end] += 1.

    # Normalize the reconstructed image by the weight map to handle overlaps
    reconstructed_image /= torch.clamp(weight_map, min=1)

    return reconstructed_image.to(patches.dtype)

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