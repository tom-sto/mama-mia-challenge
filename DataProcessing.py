import os, glob
import torch
import zarr
import numpy as np
import SimpleITK as sitk
from functools import partial
from helpers import PATCH_SIZE, NUM_PATCHES

def GetData(parentDir: str, device: torch.device):
    # assume this dir is allTheData with the 4 pre-processed data folders
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
    
    for trts in pids.keys():
        data[trts] = {}
        ids = pids[trts]
        trtsDir = os.path.join(parentDir, trts)
        for pid in ids:
            data[trts][pid] = {}
            imgs = glob.glob(os.path.join(trtsDir, f"{pid}*.nii.gz"))
            numPhases = len(imgs) - 2
            for img in imgs:
                if "seg" in img:
                    data[trts][pid]["seg"] = {i: loadImagePatches(img, device, PATCH_SIZE, NUM_PATCHES) for i in range(numPhases)}
                elif "dmap" in img:
                    data[trts][pid]["img"] = {i: loadImagePatches(img, device, PATCH_SIZE, NUM_PATCHES) for i in range(numPhases)}
                else:
                    p = int(img.split('.')[0][-1])
                    data[trts][pid]["phase"] = {p: loadImagePatches(img, device, PATCH_SIZE, NUM_PATCHES)}


    for patient_id in data["training"].keys():
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


def loadChunk(zarrArray: zarr.Array, device: torch.device, patchSize: int, chunkSize: tuple[int], startIndex: tuple[int]):
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

    patchCoords = []
    for i in range(chunkSize[0]):
        for j in range(chunkSize[1]):
            for k in range(chunkSize[2]):
                coord = (startIndex[0] + i, startIndex[1] + j, startIndex[2] + k)
                patchCoords.append(coord)

    patchCoords = torch.tensor(patchCoords).to(device)  # Shape: (numPatches, 3)
    return patches, patchCoords

# i feel very clever for this approach ;)
# Store partial handles in a data structure
# and then run them only when I need them (making sure to delete the arrays after to save memory)
def loadImagePatches(path: str, device: torch.device, patchSize: int = 32, chunkSize: tuple[int] = (4, 4, 2), overlap=2):
    assert path.endswith('.zarr'), "Only .zarr files are supported."

    n = int(overlap)
    assert np.all(np.array(chunkSize) >= n), "Need chunk size to be at least a big as expected overlap."

    zarrArray = zarr.open(path, mode='r')
    X, Y, Z = zarrArray.shape
    print(f"Image is {X} x {Y} x {Z}")
    pX = X // patchSize
    pY = Y // patchSize
    pZ = Z // patchSize

    nX = max(1, pX // (chunkSize[0] // n))
    nX = max(1, nX - (1 - pX % nX))
    nY = max(1, pY // (chunkSize[1] // n))
    nY = max(1, nY - (1 - pY % nY))
    nZ = max(1, pZ // (chunkSize[2] // n))
    nZ = max(1, nZ - (1 - pZ % nZ))
    indices = []
    for i in range(nX):
        x = min(i * chunkSize[0] / n, pX - chunkSize[0])
        x = max(x, 0)
        for j in range(nY):
            y = min(j * chunkSize[1]// n, pY - chunkSize[1])
            y = max(y, 0)
            for k in range(nZ):
                z = min(k * chunkSize[2] / n, pZ - chunkSize[2])
                z = max(z, 0)
                indices.append((int(x), int(y), int(z)))

    handles = []
    for idx in indices:
        handles.append(partial(loadChunk, zarrArray, device, patchSize, chunkSize, idx))
        
    return handles

def reconstructImageFromPatches(patches: list[torch.Tensor], patchCoords: list[list[torch.Tensor]], patchSize: int, device=None):
    """
    Reconstruct the full image from overlapping chunks using sliding window logic.
    """
    # Determine the target dimensions from the maximum coordinates in patchCoords
    max_coords = torch.max(torch.cat([coords for coords in patchCoords]), dim=0).values
    target_dims = (max_coords + 1) * patchSize
    target_dims = target_dims.tolist()

    # Initialize the reconstructed image and a weight map for averaging overlaps
    reconstructed_image: torch.Tensor = torch.zeros(target_dims, device=device, dtype=float)
    weight_map = torch.zeros(target_dims, device=device, dtype=float)

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

    handles = loadImagePatches(path, dev, patchSize=patchSize)

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