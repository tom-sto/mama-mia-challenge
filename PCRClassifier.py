import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.intermediateFeatures = 128
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.intermediateFeatures),
            nn.LayerNorm(self.intermediateFeatures),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.intermediateFeatures, 2)
        )

    def forward(self, x: torch.Tensor):
        x = x.flatten(1)
        return self.fc(x)[:,0:1]    # (B, 1)