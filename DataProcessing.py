import os, glob
import torch
import zarr
import numpy as np
import SimpleITK as sitk
from functools import partial
from helpers import PATCH_SIZE, CHUNK_SIZE

''' 
data {
    tr/val/ts: {                 ("training", "validation", "testing")
        patient_id: {       e.g. ("duke_002", "ispy2_981664")
            "seg":  [function handles] to get ground truth binary tumor segmentation + patch indices
            "dmap": [function handles] to get signed distance map of tumor segmentation + patch indices
            "phase" {
                phase_num: [function handles] to get DCE-MRI for this phase and patient + patch indices
            }
        }
    }
}
'''
def GetData(parentDir: str, device: torch.device, test: bool = False):
    print("Collecting data...")
    # assume this dir is E:\MAMA-MIA\my_preprocessed_data\Dataset106_cropped_Xch_breast_no_norm
    data = {}

    np.random.seed(420)

    pids = {}
    for trts in os.listdir(parentDir):
        trtsDir = os.path.join(parentDir, trts)
        pids[trts] = set()
        for img in os.listdir(trtsDir):
            sp = img.split("_")
            patient_id = sp[0] + "_" + sp[1]
            pids[trts].add(patient_id)
            if test and len(pids[trts]) >= 50:        # take a much smaller subset of images for testing the model so I dont have to wait 2 minutes every time
                break
    
    for trts in pids.keys():
        data[trts] = {}
        ids = pids[trts]
        trtsDir = os.path.join(parentDir, trts)
        for pid in ids:
            data[trts][pid] = {}
            data[trts][pid]["phase"] = {}
            imgs = glob.glob(os.path.join(trtsDir, f"{pid}*.zarr"))
            for img in imgs:
                if "seg" in img:
                    data[trts][pid]["seg"] = loadImagePatches(img, device, PATCH_SIZE, CHUNK_SIZE, retIndices=True)
                elif "dmap" in img:
                    data[trts][pid]["dmap"] = loadImagePatches(img, device, PATCH_SIZE, CHUNK_SIZE)
                else:
                    p = int(img.split('.')[0][-1])
                    data[trts][pid]["phase"][p] = loadImagePatches(img, device, PATCH_SIZE, CHUNK_SIZE)
    
    data["validation"] = {}
    trainingKeys = list(data["training"].keys())
    for patient_id in trainingKeys:
        if np.random.random() > 0.9:        # move 10% of training cases to validation
            data["validation"][patient_id] = data["training"][patient_id]
            del data["training"][patient_id]

    print(f"Got data: train is {len(data['training'])}, val is {len(data['validation'])}, test is {len(data['testing'])}")

    return data

def GetBinaryArrayFromTrainingTensor(tensor: torch.Tensor):
    maxIndices = torch.argmax(tensor, dim=2, keepdim=True)

    binaryTensor = torch.zeros_like(tensor)  # output tensor with same dimensions
    binaryTensor.scatter_(1, maxIndices, 1)  # write 1's into all the indices from argmax

    return binaryTensor.int()


def chunkToPatches(chunk: torch.Tensor, patchSize: int):
    # Reshape the chunk to add a batch dimension if necessary
    if chunk.dim() == 3:
        chunk = chunk.unsqueeze(0)  # Add batch dimension

    # Use unfold to extract patches along each dimension
    patches = chunk.unfold(1, patchSize, patchSize)
    patches = patches.unfold(2, patchSize, patchSize)
    patches = patches.unfold(3, patchSize, patchSize)

    # Reshape the patches to combine all dimensions into a single batch
    patches = patches.contiguous().view(-1, patchSize, patchSize, patchSize)

    return patches.to(chunk.device)


def loadChunk(zarrArray: zarr.Array, device: torch.device, patchSize: int, chunkSize: tuple[int], startIndex: tuple[int], retIndices: bool = False):
    assert len(zarrArray.shape) == 3, f"Expected zarrArray to be 3D image with no channels, got shape {zarrArray.shape}"
    X, Y, Z = zarrArray.shape
    pX = X // patchSize
    pY = Y // patchSize
    pZ = Z // patchSize

    xStart = startIndex[0] * patchSize
    yStart = startIndex[1] * patchSize
    zStart = startIndex[2] * patchSize
    xEnd   = min(xStart + chunkSize[0] * patchSize, pX * patchSize)
    yEnd   = min(yStart + chunkSize[1] * patchSize, pY * patchSize)
    zEnd   = min(zStart + chunkSize[2] * patchSize, pZ * patchSize)

    chunk = torch.tensor(zarrArray[xStart:xEnd, yStart:yEnd, zStart:zEnd]).to(device)
    patches = chunkToPatches(chunk, patchSize)

    if not retIndices: return patches

    patchIdxs = []
    for i in range(min(chunkSize[0], pX)):
        for j in range(min(chunkSize[1], pY)):
            for k in range(min(chunkSize[2], pZ)):
                coord = (startIndex[0] + i, startIndex[1] + j, startIndex[2] + k)
                # idx = getPatchIndexFromCoord(coord, pY, pZ)
                patchIdxs.append(coord)

    patchIdxs = torch.tensor(patchIdxs).to(device)  # Shape: (numPatches)

    assert len(patches) == len(patchIdxs), "Patch indexing is wrong!"
    return patches, patchIdxs

def getPatchIndexFromCoord(index: tuple[int], pY, pZ):
    return index[2] + index[1]*pZ + index[0]*pZ*pY

# i feel very clever for this approach ;)
# Store partial handles in a data structure
# and then run them only when I need them (making sure to delete the arrays after to save memory)
def loadImagePatches(path: str, device: torch.device, patchSize: int = 32, chunkSize: tuple[int] = (4, 4, 2), retIndices: bool = False):
    assert path.endswith('.zarr'), "Only .zarr files are supported."

    zarrArray = zarr.open(path, mode='r')
    try:
        X, Y, Z = zarrArray.shape
    except:
        print(f"failed on path {path}")
        return None, None
    
    # These shouldn't have any remainder since the data was padded to fit the 32 patch size!
    pX = X // patchSize         # num patches in x axis of entire image
    pY = Y // patchSize
    pZ = Z // patchSize

    chunkIndices = getChunkIndices(pX, pY, pZ, chunkSize)

    handles = [partial(loadChunk, zarrArray, device, patchSize, chunkSize, chidx, retIndices)
               for chidx in chunkIndices]
    
    return handles

def getChunkIndices(pX, pY, pZ, chunkSize):
    # if we have a chunk size greater than or equal to image, we only need one chunk
    if pX <= chunkSize[0] and pY <= chunkSize[1] and pZ <= chunkSize[2]:
        return [(0, 0, 0)]

    n = 2       # overlap chunks by #patches/2 in each direction
    assert np.all(np.array(chunkSize) >= n), "Need chunk size to be at least a big as expected overlap."
    assert np.all(np.array(chunkSize) % n == 0), f"Expected chunk size to be divisible by overlap ({n}) in all directions."

    nX = max(1, pX // (chunkSize[0] // n))      # Num chunks in x axis, overlapping by half
    nX = max(1, nX - (1 - pX % nX))             # Must be at least 1; if ^^^ has no remainder, subtract 1
    nY = max(1, pY // (chunkSize[1] // n))
    nY = max(1, nY - (1 - pY % nY))
    nZ = max(1, pZ // (chunkSize[2] // n))
    nZ = max(1, nZ - (1 - pZ % nZ))
    indices = []                                # this method is called getChunkIndices but these are the indices w.r.t. patches in the chunk
    for i in range(nX):
        x = min(i * chunkSize[0] / n, pX - chunkSize[0])    # make sure we don't index out-of-bounds! last chunk of odd-#patches images will have more overlap on this axis
        x = max(x, 0)                                       # don't use negative indices please.
        for j in range(nY):
            y = min(j * chunkSize[1] / n, pY - chunkSize[1])
            y = max(y, 0)
            for k in range(nZ):
                z = min(k * chunkSize[2] / n, pZ - chunkSize[2])
                z = max(z, 0)
                indices.append((int(x), int(y), int(z)))
    
    return indices                              # if you want actual coordinates, just multiply each index by patch size!

def reconstructImageFromPatches(patches: list[torch.Tensor], patchCoords: list[list[torch.Tensor]], patchSize: int):
    # Determine the target dimensions from the maximum coordinates in patchCoords
    max_coords = torch.max(torch.cat(patchCoords), dim=0).values
    target_dims = (max_coords + 1) * patchSize
    target_dims = target_dims.tolist()

    # Initialize the reconstructed image and a weight map for averaging overlaps
    reconstructed_image: torch.Tensor = torch.zeros(target_dims, device=patches[0].device, dtype=float)
    weight_map: torch.Tensor = torch.zeros(target_dims, device=patches[0].device, dtype=float)

    for chunk, coords in zip(patches, patchCoords):
        for patch, coord in zip(chunk, coords):
            # Calculate the slice indices for the patch
            d_start, h_start, w_start = coord * patchSize
            d_end, h_end, w_end = d_start + patchSize, h_start + patchSize, w_start + patchSize

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
    path = r"E:\MAMA-MIA\my_preprocessed_data\Dataset106_cropped_Xch_breast_no_norm\training\ispy2_738041_seg.zarr"
    outFolder = os.path.dirname(os.path.dirname(path))
    arr = zarr.load(path)
    # print(arr.dtype)
    if "seg" in path:
        arr = arr.astype(int)
    nii = sitk.GetImageFromArray(arr)
    sitk.WriteImage(nii, os.path.join(outFolder, "og.nii"))
    print("Wrote og.nii")

    patchSize = 32
    dev = torch.device("cuda")

    handles, indices = loadImagePatches(path, dev, patchSize=patchSize)

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
    if "seg" in path:
        recon = recon.int()

    sitk.WriteImage(sitk.GetImageFromArray(recon.cpu().detach().numpy()), os.path.join(outFolder, "recon.nii"))