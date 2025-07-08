import torch
import torch.nn as nn
import numpy as np

# copied architecture from segmentation autoencoder (Tiramisu 2.0)
class MyRefiner(torch.nn.Module):
    def __init__(self,
                 down_channels: list[int] = [3, 8, 32, 64, 96],
                 up_channels: list[int] = [2, 8, 32, 64, 96],      # end with 2 channels so nnUNet trainer can do cross-entropy loss
                 n_bottleneck_layers: int = 3
    ):
        super(MyRefiner, self).__init__()
        assert len(down_channels) == len(up_channels), 'Need up channels and down channels to be the same length. \
                                                        This is a symmetric network!'
        encoderSteps    = nn.ModuleList([])
        bottleneckSteps = nn.ModuleList([])
        decoderSteps    = nn.ModuleList([])

        num_blocks = len(down_channels) - 1

        # Populate the encoder and decoder modules symmetrically (Order is important here!)
        for i in range(num_blocks):
            # if this is the last block before the latent space, add dropout
            if i == num_blocks - 1:
                encoderSteps.append(nn.Dropout3d(0.2))

            # Antonio says to put BatchNorm first in each encoder block
            encoderSteps.append(nn.BatchNorm3d(num_features=down_channels[i]))

            # two consecutive 3x3x3 convolutions in first three encoder blocks
            if i == 0 or i == 1 or i == 2:
                encoderSteps.append(nn.Conv3d(in_channels=down_channels[i], out_channels=down_channels[i], kernel_size=3, padding=1))
            encoderSteps.append(nn.Conv3d(in_channels=down_channels[i], out_channels=down_channels[i+1], kernel_size=3, padding=1))
            encoderSteps.append(nn.ReLU(True))
            if i % 2:
                encoderSteps.append(nn.Dropout3d(0.1))
            encoderSteps.append(nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
            
            # don't run Batch Norm or ReLU on the last decoder block, just send conv output straight to loss fn
            # this check went from outputs with 2 missing segments to all segments looking good
            # we also want to make sure that we don't activate before the loss is calculated
            if i != 0:
                decoderSteps.insert(0, nn.ReLU(True))   # Note that we are inserting at the front! These get put in reverse order
                if i % 2:
                    decoderSteps.insert(0, nn.Dropout3d(0.1))
                decoderSteps.insert(0, nn.BatchNorm3d(num_features=up_channels[i]))
            decoderSteps.insert(0, nn.Conv3d(in_channels=up_channels[i] + down_channels[i+1], out_channels=up_channels[i], kernel_size=3, padding=1))
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
        x = x.to(torch.float32)
        encoder_outputs = []

        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool3d):
                encoder_outputs.append(x)           # take skips from right after activation
            x = layer(x)

        # for i, skip in enumerate(encoder_outputs):
        #     print(f"Skip {i} has shape {skip.shape}")

        # output here is the latent space
        for layer in self.bottleneck:
            x = layer(x)
        
        self.latentRep = torch.flatten(x, start_dim=1)      # keep images in batch separate

        skip_idx = len(encoder_outputs) - 1
        for layer in self.decoder:
            if isinstance(layer, nn.Conv3d):        # add skips in front of Conv3d
                skip = encoder_outputs[skip_idx]
                # print(f"Decoder {i}: trying to take skip of shape {skip.shape}")
                # print(f"\t and append it to x: {x.shape}")
                x = torch.cat([x, skip], dim=1)
                # print(f"\tNew x shape: {x.shape}")
                skip_idx -= 1
            x = layer(x)
        
        return x