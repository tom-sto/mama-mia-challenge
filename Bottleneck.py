import torch
import torch.nn as nn
from AttentionPooling import AttentionPooling

class NoBottleneck(nn.Module):
    def __init__(self, channelsIn, nHeads):
        super().__init__()

        self.pool = AttentionPooling(channelsIn, nHeads)

    # x comes in with shape (B, N*X*Y*Z, T, E)
    def forward(self, x: torch.Tensor, shape: tuple[int], *_):
        B, T, N, E, X, Y, Z = shape

        x = x.reshape(B, N, X, Y, Z, T, E)
        x = x.permute(0, 1, 5, 6, 2, 3, 4)          # (B, N, T, E, X, Y, Z)

        # reduce T to 1 with attention pooling
        x = x.reshape(B, N, T, E*X*Y*Z)
        x = self.pool(x)                            # (B, N, E*X*Y*Z)
        x = x.reshape(B, N, E, X, Y, Z)

        return x
    
class ConvBottleneck(nn.Module):
    def __init__(self, channelsIn: int, channelsOut: int, nHeads: int, nLayers: int):
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

        self.Eout = channelsOut
        self.pool = AttentionPooling(self.Eout, nHeads)

    # x comes in with shape (B, N*X*Y*Z, T, E)
    def forward(self, x: torch.Tensor, shape: tuple[int], *_):
        B, T, N, E, X, Y, Z = shape

        # need x to be ready for convolutions: (B*T*N, E, X, Y, Z)
        x = x.reshape(B, N, X, Y, Z, T, E)
        x = x.permute(0, 1, 5, 6, 2, 3, 4)          # (B, N, T, E, X, Y, Z)
        x = x.reshape(-1, E, X, Y, Z)               # (B*N*T, E, X, Y, Z)
        x = self.bottleneckSteps(x)                 # (B*N*T, E', X, Y, Z)

        # reduce T to 1 with attention pooling
        x = x.reshape(B, N, T, self.Eout*X*Y*Z)
        x = self.pool(x)                            # (B, N, E'*X*Y*Z)
        x = x.reshape(B, N, self.Eout, X, Y, Z)

        return x