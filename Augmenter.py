import torch
import torchio as tio

def GetTrainingTransforms():
    transform = tio.Compose([
        tio.RandomAffine(scales=(0.8, 1.1), degrees=15, translation=(5, 5, 5), p=0.8),
        tio.RandomGamma(log_gamma=(0.9, 1.1), p=0.5),
        tio.RandomNoise(mean=0, std=0.01, p=0.5),
    ])
    return transform

def GetNoTransforms():
    return tio.Compose([tio.RandomAffine(p=0)])

def DoTransforms(phases: torch.Tensor, dmap: torch.Tensor, seg: torch.Tensor, transform: tio.Compose):
    subject_dict = {}

    subject_dict[f'phases'] = tio.ScalarImage(tensor=phases)
    
    if dmap is not None:
        subject_dict['labels'] = tio.LabelMap(tensor=torch.stack([dmap, seg]))
    else:
        subject_dict['labels'] = tio.LabelMap(tensor=seg.unsqueeze(0))
    
    subject = tio.Subject(subject_dict)
    transformed = transform(subject)

    phases = transformed['phases'].data.squeeze()
    labels = transformed['labels'].data
    if dmap is not None: 
        dmap = labels[0]
        seg  = labels[1]
    else:
        seg = labels.squeeze()

    return phases, dmap, seg
