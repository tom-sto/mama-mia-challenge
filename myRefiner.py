import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import os
from torch.utils.data import DataLoader
from corruptor import ReadBinaryArrayFromFile

# copied architecture from segmentation autoencoder (Tiramisu 2.0)
class MyRefiner(torch.nn.Module):
    def __init__(self,
                 dropout_p: float = 0.2,
                 down_channels: list[int] = [1, 8, 32, 64, 128],
                 up_channels: list[int] = [2, 8, 32, 64, 128],      # end with 2 channels so nnUNet trainer can do cross-entropy loss
                 n_bottleneck_layers: int = 3
    ):
        super(MyRefiner, self).__init__()
        assert len(down_channels) == len(up_channels), 'Need up channels and down channels to be the same length. \
                                                        This is a symmetric network!'
        encoderSteps    = nn.ModuleList([])
        bottleneckSteps = nn.ModuleList([])
        decoderSteps    = nn.ModuleList([])

        # learnable threshold for final reconstruction of binary segmentation from logits
        num_blocks = len(down_channels) - 1

        # Populate the encoder and decoder modules symmetrically (Order is important here!)
        for i in range(num_blocks):
            # if this is the last block before the latent space, add dropout
            if i == num_blocks - 1:
                encoderSteps.append(nn.Dropout3d(dropout_p))

            # Antonio says to put BatchNorm first in each encoder block
            encoderSteps.append(nn.BatchNorm3d(num_features=down_channels[i]))

            # two consecutive 3x3x3 convolutions in first two encoder blocks
            if i == 0 or i == 1:
                encoderSteps.append(nn.Conv3d(in_channels=down_channels[i], out_channels=down_channels[i], kernel_size=3, padding=1))
            encoderSteps.append(nn.Conv3d(in_channels=down_channels[i], out_channels=down_channels[i+1], kernel_size=3, padding=1))
            encoderSteps.append(nn.ReLU(True))
            encoderSteps.append(nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
            
            # don't run Batch Norm or ReLU on the last decoder block, just send conv output straight to loss fn
            # this check went from outputs with 2 missing segments to all segments looking good
            # we also want to make sure that we don't activate before the loss is calculated
            if i != 0:
                decoderSteps.insert(0, nn.ReLU(True))   # Note that we are inserting at the front! These get put in reverse order
                decoderSteps.insert(0, nn.BatchNorm3d(num_features=up_channels[i]))
            decoderSteps.insert(0, nn.Conv3d(in_channels=up_channels[i], out_channels=up_channels[i], kernel_size=3, padding=1))
            decoderSteps.insert(0, nn.ConvTranspose3d(in_channels=up_channels[i+1], 
                                                      out_channels=up_channels[i], 
                                                      kernel_size=3, 
                                                      stride=2, 
                                                      padding=1, 
                                                      output_padding=1)) # need padding here to make sure spatial dimensions get decoded correctly
        
        # Bottleneck layers
        # Ensures we always get the right number of channels in our latent space
        bottleneckChannels = np.linspace(down_channels[-1], up_channels[-1], n_bottleneck_layers + 1, dtype=np.int_)
        for i in range(n_bottleneck_layers):
            bottleneckSteps.append(nn.Conv3d(in_channels=bottleneckChannels[i], out_channels=bottleneckChannels[i+1], kernel_size=3, padding=1))

        self.encoder    = encoderSteps
        self.decoder    = decoderSteps
        self.bottleneck = bottleneckSteps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)

        # output here is the latent space
        for layer in self.bottleneck:
            x = layer(x)
        
        self.latentRep = torch.flatten(x, start_dim=1)      # keep images in batch separate

        for layer in self.decoder:
            x = layer(x)
        
        return x
    
        soft_mask = torch.sigmoid(100 * (x - self.threshold))

        return soft_mask
    
# # make a dataloader that takes a filename, reads the binary array from file, and gets the ground truth from E:\MAMA-MIA\segmentations\expert
# class MyRefinerDataset(torch.utils.data.Dataset):
#     def __init__(self, file_paths: list[str], ground_truth_dir: str):
#         self.file_paths = file_paths
#         self.ground_truth_dir = ground_truth_dir

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx: int):
#         file_path = self.file_paths[idx]
#         binary_array = ReadBinaryArrayFromFile(file_path)
        
#         # Get the patient ID from the file name
#         patient_id = file_path.split('/')[-1].split('.')[0][:-2]    # remove the augmentation id
#         ground_truth_path = f"{self.ground_truth_dir}/{patient_id}.nii.gz"
#         ground_truth_image = sitk.ReadImage(ground_truth_path)
        
#         # Convert SimpleITK image to numpy array and then to tensor
#         ground_truth_array = sitk.GetArrayFromImage(ground_truth_image)
#         ground_truth_tensor = torch.from_numpy(ground_truth_array, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
#         return torch.from_numpy(binary_array, dtype=torch.float32).unsqueeze(0), ground_truth_tensor