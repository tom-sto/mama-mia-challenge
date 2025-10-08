import torch
import torchio as tio

def GetTrainingTransforms():
    transform = tio.Compose([
        tio.RandomAffineElasticDeformation(
            affine_first=True, 
            affine_kwargs={
                "scales": (0.8, 1.1),
                "degrees": 15,
                "translation": (2, 2, 2),
                "p": 0.8
            },
            elastic_kwargs={
                "max_displacement": 5,
                "p": 0.2
            }
        ),
        tio.RandomGamma(log_gamma=(0.85, 1.15), p=0.5),
        tio.RandomNoise(mean=0, std=0.01, p=0.5),
        tio.RandomBlur(std=(0.5, 1.0), p=0.5),
    ])
    return transform

def GetNoTransforms():
    return tio.Compose([tio.RandomAffine(p=0)])

def DoTransforms(phases: list[torch.Tensor], seg: torch.Tensor, dmap: torch.Tensor, transform: tio.Compose):
    subject_dict = {}

    for i, img in enumerate(phases):
        while img.ndim < 4:
            img = img.unsqueeze(0)
        subject_dict[f'phase_{i}'] = tio.ScalarImage(tensor=img)
    while seg.ndim < 4:
        seg = seg.unsqueeze(0)
    subject_dict['segmentation'] = tio.LabelMap(tensor=seg)
    if dmap is not None:
        while dmap.ndim < 4:
            dmap = dmap.unsqueeze(0)
        subject_dict['distance_map'] = tio.LabelMap(tensor=dmap)
    
    subject = tio.Subject(subject_dict)
    transformed = transform(subject)

    transformed_images = [transformed[f'phase_{i}'].data.squeeze() for i in range(len(phases))]
    transformed_segmentation = transformed['segmentation'].data.squeeze()
    transformed_distance_map = None
    if dmap is not None: 
        transformed_distance_map = transformed['distance_map'].data.squeeze()

    return transformed_images, transformed_segmentation, transformed_distance_map
