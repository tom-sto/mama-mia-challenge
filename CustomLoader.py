import random
import torch
from DataProcessing import GetData
from torch.utils.data import Dataset, Sampler

# need this to load and track training data along with dmap vectors
# group tensors by patient ID 
# (then image type: MRI, dmap, seg)
# then by phase (training data has between 3 and 6 phases)
class CustomDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __getitem__(self, ind: str):
        patientID = ind
        handle = self.data[patientID]
                        
        return handle, patientID
    
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

# TODO: Make batch sampler that takes patients with matching # phases
def GetDataloaders(dataDir: str, device: torch.device, oversample: float, batchSize: int = 1, shuffle=True, test=False):
    my_collate = lambda x: x[0]         # unpack from singleton list
    data = GetData(dataDir, device, oversample, test=test)
    trDataset = CustomDataset(data=data["training"])
    trPIDs = list(data["training"].keys())
    trSampler = CustomSampler(trPIDs, shuffle=shuffle)
    trDataloader = torch.utils.data.DataLoader(trDataset, batch_size=batchSize, sampler=trSampler, collate_fn=my_collate)
    
    vlDataset = CustomDataset(data=data["validation"])
    vlPIDs = list(data["validation"].keys())
    vlSampler = CustomSampler(vlPIDs, shuffle=shuffle)
    vlDataloader = torch.utils.data.DataLoader(vlDataset, batch_size=batchSize, sampler=vlSampler, collate_fn=my_collate)
    
    tsDataset = CustomDataset(data=data["testing"])
    tsPIDs = list(data["testing"].keys())
    tsSampler = CustomSampler(tsPIDs, shuffle=shuffle)
    tsDataloader = torch.utils.data.DataLoader(tsDataset, batch_size=batchSize, sampler=tsSampler, collate_fn=my_collate)
    
    return trDataloader, vlDataloader, tsDataloader