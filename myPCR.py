import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=2, out_features=128):
        super(Simple3DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),  # [B, 32, D/2, H/2, W/2]

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(2),  # [B, 64, D/4, H/4, W/4]

            nn.Conv3d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(2),  # [B, 96, D/8, H/8, W/8]

            nn.Conv3d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.AdaptiveAvgPool3d(1)  # [B, 128, 1, 1, 1]
        )
        self.out_features = out_features
        self.fc = nn.Linear(128, out_features)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x  # [B, out_features]


class MetadataMLP(nn.Module):
    def __init__(self, device, in_dim=7, out_dim=32):
        super(MetadataMLP, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, out_dim)
        )

        ageMin = 21
        ageMax = 77
        self.ageEncode  = lambda x: torch.tensor([(x - ageMin) / (ageMax - ageMin)], dtype=torch.float32) if x is not None \
            else torch.tensor([0.5], dtype=torch.float32)   # normalize age to [0, 1] range, default to 0.5 if None
        self.menoEncode = lambda x: torch.tensor([1., 0.]) if x == "pre" \
            else torch.tensor([0., 1.]) if x == "post" \
            else torch.tensor([0., 0.])
        self.densityEncode = lambda x: torch.tensor([1., 0., 0., 0.]) if x == "a" \
            else torch.tensor([0., 1., 0., 0.]) if x == "b" \
            else torch.tensor([0., 0., 1., 0.]) if x == "c" \
            else torch.tensor([0., 0., 0., 1.]) if x == "d" \
            else torch.tensor([0., 0., 0., 0.])

    def forward(self, metadata: list[dict]):
        metadata_emb = torch.zeros(len(metadata), self.in_dim, device=self.device)

        for idx, md in enumerate(metadata):
            age = md['age']
            menopausal_status = md['menopausal_status']
            breast_density = md['breast_density']
            age_emb = self.ageEncode(age).to(self.device)
            meno_emb = self.menoEncode(menopausal_status).to(self.device)
            density_emb = self.densityEncode(breast_density).to(self.device)
            concat = torch.cat((age_emb, meno_emb, density_emb), dim=0).unsqueeze(0)
            metadata_emb[idx] = concat
        
        return self.net(metadata_emb)


class PCRPredictor(nn.Module):
    def __init__(self, device, use_metadata=True):
        super(PCRPredictor, self).__init__()
        self.use_metadata = use_metadata

        self.image_encoder = Simple3DCNN(in_channels=4, out_features=128)

        if self.use_metadata:
            self.metadata_encoder = MetadataMLP(device)
            fusion_dim = 128 + 32
        else:
            fusion_dim = 128

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, metadata=None):
        """
        Inputs:
        - x:        [B, 4, D, H, W]
        - metadata: [B, metadata_dim] or None

        Returns:
        - pCR probability: [B, 1]
        """
        print("x is type", x.type())
        x = x.to(torch.float32)
        img_feat = self.image_encoder(x)

        if self.use_metadata and metadata is not None:
            meta_feat = self.metadata_encoder(metadata)
            fused = torch.cat([img_feat, meta_feat], dim=1)
        else:
            fused = img_feat

        return self.classifier(fused)
