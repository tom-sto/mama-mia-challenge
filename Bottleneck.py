import torch
import torch.nn as nn

class NoBottleneck(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x
    
class ConvBottleneck(nn.Module):
    def __init__(self, channelsIn: int, channelsOut: int, nLayers: int):
        super().__init__()
        bottleneckChannels = torch.linspace(channelsIn, 
                                            channelsOut, 
                                            nLayers + 1, 
                                            dtype=int)
        self.bottleneckSteps = nn.Sequential(*
            [nn.Conv3d(bottleneckChannels[i],
                       bottleneckChannels[i+1], 
                       kernel_size=3, 
                       padding=1)
            for i in range(nLayers)]
        )

    def forward(self, x: torch.Tensor):
        return self.bottleneckSteps(x)