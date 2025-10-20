import random
import torch
from DataProcessing import GetData
from torch.utils.data import Dataset, BatchSampler
from helpers import DTYPE_DMAP, DTYPE_SEG, DTYPE_PHASE, DTYPE_PCR
from Augmenter import DoTransforms

# need this to load and track training data along with dmap vectors
# group tensors by patient ID 
# (then image type: MRI, dmap, seg)
# then by phase (training data has between 3 and 6 phases)
class CustomDataset(Dataset):
    def __init__(self, data: dict, augmentCompose):
        self.data = data
        self.augment = augmentCompose

    def __getitem__(self, ind: str):
        patientID, numPhases = ind
        handle, pcr = self.data[numPhases][patientID]
        phaseArrs, dmapArr, segArr, bbox = handle()         # this is where we actually load data from disk
        # ^^^ this is good because it lets torch handle pre-loading image with persistant workers

        phaseTensors = torch.from_numpy(phaseArrs).to(DTYPE_PHASE)
        phaseTensors = phaseTensors / phaseTensors.amax(dim=(1, 2, 3), keepdim=True)       # normalize each phase intensity here
        dmapTensors  = torch.from_numpy(dmapArr).to(DTYPE_DMAP) if dmapArr is not None else None
        segTensors   = torch.from_numpy(segArr).to(DTYPE_SEG)
        phase, dmap, seg = DoTransforms(phaseTensors, 
                                        dmapTensors, 
                                        segTensors,
                                        self.augment)
        pcr = torch.tensor(pcr).to(DTYPE_PCR)
        return phase, dmap, seg, pcr, bbox, patientID
    
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

def GetDataloaders(dataDir: str, patientDataPath: str, trAugCompose, vlTsAugCompose,
                   downsampleFactor: int = 1, batchSize: int = 1, shuffle=True, test=False):
    data = GetData(dataDir, patientDataPath, downsampleFactor, test=test)

    def makeLoader(split: str, augmentCompose):
        dataset = CustomDataset(data=data[split], augmentCompose=augmentCompose)
        sampler = CustomBatchSampler(data[split], batchSize=1 if split!="training" else batchSize, shuffle=shuffle)
        return torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, 
                                           num_workers=min(batchSize, 12), pin_memory=True, persistent_workers=True)

    return makeLoader("training", trAugCompose), makeLoader("validation", vlTsAugCompose), makeLoader("testing", vlTsAugCompose)