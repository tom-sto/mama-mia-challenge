import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.intermediateFeatures = [128, 64]
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            # nn.Dropout(0),
            nn.Linear(dim, self.intermediateFeatures[0]),
            nn.LayerNorm(self.intermediateFeatures[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.intermediateFeatures[0], self.intermediateFeatures[1]),
            nn.LayerNorm(self.intermediateFeatures[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.intermediateFeatures[1], 2)
        )

    def forward(self, x: torch.Tensor):
        x = x.flatten(1)
        return self.fc(x)[:,0:1]    # (B, 1)