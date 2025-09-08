import os, glob
import torch
import zarr, math
import numpy as np
import SimpleITK as sitk
from functools import partial
from itertools import product
from helpers import PATCH_SIZE, NUM_PATCHES, DTYPE_SEG, DTYPE_DMAP, DTYPE_PHASE

''' 
data {
    tr/val/ts: {                 ("training", "validation", "testing")
        patient_id:         e.g. ("duke_002", "ispy2_981664")
            handle - partial that returns
                - [segImg tensor]       (should be 1)
                - [dmapImg tensor]      (should be 1)
                - [phaseImg tensor]     (should be 2-5)
                - [patchIndices]
    }
}
'''
def GetData(parentDir: str, device: torch.device, oversample: float = 0., test: bool = False):
    print("Collecting data...")
    # assume this dir is E:\MAMA-MIA\my_preprocessed_data\Dataset106_cropped_Xch_breast_no_norm
    data = {}

    np.random.seed(420)

    pids = {}
    for trts in os.listdir(parentDir):
        trtsDir = os.path.join(parentDir, trts)
        if not os.path.isdir(trtsDir):
            continue
        pids[trts] = set()
        for img in os.listdir(trtsDir):
            sp = img.split("_")
            patient_id = sp[0] + "_" + sp[1]
            pids[trts].add(patient_id)

    if test:
        pids["training"] = ["duke_612", "duke_778", "ispy2_255078", "ispy2_410083"]
        pids["testing"] = ["duke_385"]
    
    for trts in pids.keys():
        data[trts] = {}
        ids = pids[trts]
        trtsDir = os.path.join(parentDir, trts)
        for pid in ids:
            imgs = glob.glob(os.path.join(trtsDir, f"{pid}*.zarr"))
            bboxPath = os.path.join(trtsDir, f"{pid}_bbox.txt")
            with open(bboxPath, 'r') as f:
                bbox = eval(f.read().strip())
            
            handle = loadImagePatches(imgs, device, PATCH_SIZE, NUM_PATCHES,
                                       oversample=oversample, fgBox=bbox,
                                       loadWholeImage=(trts=="testing"))
            data[trts][pid] = handle
    
    data["validation"] = {}
    trainingKeys = list(data["training"].keys())
    for patient_id in trainingKeys:
        if np.random.random() > 0.9:        # move 10% of training cases to validation
            data["validation"][patient_id] = data["training"][patient_id]
            del data["training"][patient_id]

    if test:
        data["validation"] = {}
        data["validation"]["duke_612"] = data["training"]["duke_612"]
        del data["training"]["duke_612"]

    print(f"Got data: train is {len(data['training'])}, val is {len(data['validation'])}, test is {len(data['testing'])}")

    return data

def GetBinaryArrayFromTrainingTensor(tensor: torch.Tensor):
    maxIndices = torch.argmax(tensor, dim=2, keepdim=True)

    binaryTensor = torch.zeros_like(tensor)  # output tensor with same dimensions
    binaryTensor.scatter_(1, maxIndices, 1)  # write 1's into all the indices from argmax

    return binaryTensor.int()

# need to select patch indices inside of function handle since we want it to be consistent for all images in a case,
# but it should pick different patches each iteration
def getPatches(segPaths: list[str], dmapPaths: list[str], phasePaths: list[str], device: torch.device, patchSize: int,
               numPatches: int, oversample: float, fgBox: tuple[list[int]], loadWholeImage: bool):
    shape = zarr.open(segPaths[0], mode='r').shape
    patchIndices = getIndices(numPatches, patchSize, shape, 
                              oversample=oversample, fgBox=fgBox, 
                              loadWholeImage=loadWholeImage)
    
    # I know this is not very readable, but all we're doing is looping over opened Zarr arrays from the paths
    # then loading each patch from the randomly generated indices for each path.
    # segImgs and dmapImgs should only have 1 image each, but it hurts my brain less to just do them all the same way
    segImgs     = torch.tensor(
                    np.array(
                        [[zarrArray[idx[0]:idx[0] + patchSize, idx[1]:idx[1] + patchSize, idx[2]:idx[2] + patchSize] for idx in patchIndices] 
                            for zarrArray in [zarr.open(path, mode='r') for path in segPaths]]
                    )).to(device, dtype=DTYPE_SEG)
    dmapImgs    = torch.tensor(
                    np.array(
                        [[zarrArray[idx[0]:idx[0] + patchSize, idx[1]:idx[1] + patchSize, idx[2]:idx[2] + patchSize] for idx in patchIndices] 
                            for zarrArray in [zarr.open(path, mode='r') for path in dmapPaths]]
                    )).to(device, dtype=DTYPE_DMAP)
    phaseImgs   = torch.tensor(
                    np.array(
                        [[zarrArray[idx[0]:idx[0] + patchSize, idx[1]:idx[1] + patchSize, idx[2]:idx[2] + patchSize] for idx in patchIndices] 
                            for zarrArray in [zarr.open(path, mode='r') for path in phasePaths]]
                    )).to(device, dtype=DTYPE_PHASE)
    
    # TODO: Finish me!
    pcrValues   = [None]
    
    return segImgs, dmapImgs, phaseImgs, pcrValues, patchIndices

def getPatchIndexFromCoord(index: tuple[int], pY, pZ):
    return index[2] + index[1]*pZ + index[0]*pZ*pY

# i feel very clever for this approach ;)
# Store partial handles in a data structure
# and then run them only when I need them (making sure to delete the arrays after to save memory)
# return segImg handles, dmapImg handles, [phaseImg handles for p in nPhases] 
def loadImagePatches(paths: list[str], device: torch.device, patchSize: int, 
                     numPatches: int, oversample: float, fgBox: tuple[list[int]],
                     loadWholeImage: bool = False):
    assert all(path.endswith('.zarr') for path in paths), "Only .zarr files are supported."

    segPaths    = [p for p in paths if "seg" in p]
    dmapPaths   = [p for p in paths if "dmap" in p]
    phasePaths  = [p for p in paths if "seg" not in p and "dmap" not in p]
    
    # need images in ascending order!
    segPaths.sort()
    dmapPaths.sort()
    phasePaths.sort()

    return partial(getPatches, segPaths, dmapPaths, phasePaths, device, patchSize, numPatches, oversample, fgBox, loadWholeImage)

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
            # sample the start index from 
            r1 = range(minFG[0] - patchSize // 2, maxFG[0] - patchSize // 2)
            r2 = range(minFG[1] - patchSize // 2, maxFG[1] - patchSize // 2)
            r3 = range(minFG[2] - patchSize // 2, maxFG[2] - patchSize // 2)
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

# this function undoes subpatching, returning a tensor to its original size
# (B*N, C, P, P, P) -> (B, C, X, Y, Z)
# X == Y == Z
# N == numSubpatches
def unpatchTensor(x: torch.Tensor, numSubpatches: int):
    assert len(x.shape) == 5, f"Expected shape (B*N, C, P, P, P). Got {x.shape}"
    B_N, C, P, _, _ = x.shape
    numSubpatchesPerDim = round(numSubpatches ** (1/3))
    X = Y = Z = P * numSubpatchesPerDim

    B = B_N // numSubpatches

    # Reshape back to original dimensions
    x = x.view(B, numSubpatchesPerDim, numSubpatchesPerDim, numSubpatchesPerDim, C, P, P, P)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)  # [B, C, X//P, P, Y//P, P, Z//P, P]
    x = x.reshape(B, C, X, Y, Z)  # [B, C, X, Y, Z]

    return x

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