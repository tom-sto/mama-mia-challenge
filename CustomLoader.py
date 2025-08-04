import random
import torch
import numpy as np
import zarr
from torch.utils.data import Dataset, Sampler
from helpers import IMAGE_TYPES
from typing import Optional, Union

# need this to load and track training data along with dmap vectors
# group tensors by patient ID 
# (then image type: MRI, dmap, seg)
# then by phase (training data has between 3 and 6 phases)
class CustomDataset(Dataset):
    def __init__(self, data: dict, device: torch.device):
        self.data = data
        self.device = device

    def __len__(self):
        axis1 = len(self.data.values())
        x = list(self.data.keys())[0]
        axis2 = len(self.data[x].values())
        return axis1*axis2

    def __getitem__(self, ind: tuple[str, int, int]):
        # unpack id and transform from index
        patientID, phase, patchIdx = ind
        
        # need MRIs, dmap, and seg
        images: dict[str, torch.Tensor] = {}
        for imageType in IMAGE_TYPES:
            if not imageType in self.data[patientID]:
                continue

            handle = self.data[patientID][imageType][phase]
            # load data here so we don't have to put all 40GB into RAM at once
            data: np.ndarray = handle[patchIdx]()
            # idk this is from Jiawei's data loader
            if imageType == "ct" and len(data.shape)==3:
                data = np.expand_dims(data, axis=0)
            if imageType == "latent_heatmap":
                data = data.reshape(LATENT_LMK_SHAPE)
            if imageType == "latent_segmentation":
                data = data.reshape(LATENT_SEG_SHAPE)
            images[imageType] = torch.from_numpy(data)

        return images, patientID, phase
    
class CustomSampler(Sampler):
    def __init__(self, ind, shuffle=False, seed=None):
        super().__init__()
        self.indices = ind
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # Set the seed for reproducibility if specified
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
def GetDatasetAndSampler(data: dict, forAutoencoder: bool, device: torch.device, numChunks: int, numTransforms: int=5, shuffle=True, collect="both"):
    match collect:
        case "both":
            dataset = CustomDataset(data=data, forAutoencoder=forAutoencoder, device=device)
            IDs = list(data.keys())
        case "cranio":
            cranioData = {k: v for k, v in data.items() if "Patient" in k}
            dataset = CustomDataset(data=cranioData, forAutoencoder=forAutoencoder, device=device)
            IDs = list(cranioData.keys())
        case "norm":
            normData = {k: v for k, v in data.items() if "Patient" not in k}
            dataset = CustomDataset(data=normData, forAutoencoder=forAutoencoder, device=device)
            IDs = list(normData.keys())
        case _:
            print("Invalid argument for collect:", collect)
            return
    
    transforms = list(range(numTransforms))
    patchIndexes = list(range(numChunks))
    indexes = [(ID, transform, patchIdx) for ID in IDs for transform in transforms for patchIdx in patchIndexes]
    sampler = CustomSampler(indexes, shuffle=shuffle)
    return dataset, sampler