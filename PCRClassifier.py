import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor):
        x = x.flatten(1)
        return self.fc(x) # (B, 1)

# class ClassifierHead(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(dim, 384),
#             nn.ReLU(),
#             nn.LayerNorm(384),
#             nn.Dropout(p=0.2),
#             nn.Linear(384, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )

#     def forward(self, x: torch.Tensor):
#         x = x.flatten(1)
#         return self.fc(x) # (B, 1)
    
class ClassifierHeadWithConfidence(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, 384),
            nn.ReLU(),
            nn.LayerNorm(384),
            nn.Dropout(p=0.2),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x: torch.Tensor):
        x = x.flatten(1)
        out = self.fc(x)  # (B, 2)
        logit = out[:, 0]         # (B, 1)
        logitConf = out[:, 1]    # (B, 1)
        return logit, logitConf