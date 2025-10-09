import random
import torch
from DataProcessing import GetData
from torch.utils.data import Dataset, BatchSampler

# need this to load and track training data along with dmap vectors
# group tensors by patient ID 
# (then image type: MRI, dmap, seg)
# then by phase (training data has between 3 and 6 phases)
class CustomDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __getitem__(self, ind: str):
        patientID, numPhases = ind
        handles = self.data[numPhases][patientID]
                        
        return handles, patientID
    
class CustomBatchSampler(BatchSampler):
    def __init__(self, dataSplit: dict, batchSize: int, shuffle: bool = False, dropLast: bool = False, seed=None):
        self.dataSplit = dataSplit
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.dropLast = dropLast
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        batches = []

        for numPhases, patients in self.dataSplit.items():
            patientIDs = list(patients.keys())

            if self.shuffle:
                rng.shuffle(patientIDs)

            # I'm gonna cheat a little bit here
            # When we load images with more phases, we use more data (obviously).
            # So use smaller batch size with higher phase counts so it fits in VRAM
            if numPhases > 4:
                batchSize = max(1, self.batchSize // 2)
            else:
                batchSize = self.batchSize
            
            for i in range(0, len(patientIDs), batchSize):
                batchPids = patientIDs[i:i + batchSize]
                batch = [(pid, numPhases) for pid in batchPids]
                if self.dropLast and len(batch) < batchSize:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        total = 0
        for numPhases, patients in self.dataSplit.items():
            if numPhases == 5 or numPhases == 6:
                batchSize = max(1, self.batchSize // 2)
            else:
                batchSize = self.batchSize
            
            n = len(patients)
            if self.dropLast:
                total += n // batchSize
            else:
                total += (n + batchSize - 1) // batchSize
        return total

# need this because Torch doesn't handle loading data from dictionaries automatically
def collate(x):
    return x

def GetDataloaders(dataDir: str, patientDataPath: str, batchSize: int = 1, shuffle=True, test=False):
    data = GetData(dataDir, patientDataPath, test=test)

    def makeLoader(split: str):
        dataset = CustomDataset(data=data[split])
        sampler = CustomBatchSampler(data[split], batchSize=1 if split!="training" else batchSize, shuffle=shuffle)
        return torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, 
                                           num_workers=4, pin_memory=True, persistent_workers=True)

    return makeLoader("training"), makeLoader("validation"), makeLoader("testing")