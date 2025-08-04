import torch
import torch.nn as nn

def subpatchTensor(x: torch.Tensor, subpatchSize: int):
    assert len(x.shape) == 5, f"Expected shape (B, C, X, Y, Z). Got: {x.shape}"
    B, T, X, Y, Z = x.shape
    assert X % subpatchSize == 0 and Y % subpatchSize == 0 and Z % subpatchSize == 0, \
        "Input dimensions must be divisible by subpatchSize"

    # Reshape into subpatches
    x = x.unfold(2, subpatchSize, subpatchSize)  # Unfold X
    x = x.unfold(3, subpatchSize, subpatchSize)  # Unfold Y
    x = x.unfold(4, subpatchSize, subpatchSize)  # Unfold Z
    x = x.permute(0, 2, 3, 4, 1, 5, 6, 7)  # [B, X//subpatchSize, Y//subpatchSize, Z//subpatchSize, C, subpatchSize, subpatchSize, subpatchSize]
    x = x.reshape(B, -1, T, subpatchSize, subpatchSize, subpatchSize)  # [B, N, T, subpatchSize, subpatchSize, subpatchSize]
    x = x.unsqueeze(3)          # Add a singleton channel dimension for patch embedding: [B, N, T, 1, subpatchSize, subpatchSize, subpatchSize]
    x = x.permute(0, 2, 1, 3, 4, 5, 6)      # [B, T, N, 1, subpatchSize, subpatchSize, subpatchSize]

    numSubpatches = (X // subpatchSize) * (Y // subpatchSize) * (Z // subpatchSize)
    return x, numSubpatches

# this function undoes subpatching, returning a tensor to its original size
# (B*N, C, P, P, P) -> (B, C, X, Y, Z)
# X == Y == Z
# N == numSubpatches
def unpatchTensor(x: torch.Tensor, numSubpatches: int):
    assert len(x.shape) == 5, f"Expected shape (B*N, C, P, P, P). Got {x.shape}"
    B_N, C, P, _, _ = x.shape
    numSubpatchesPerDim = round(numSubpatches ** (1/3))
    X = Y = Z = P * numSubpatchesPerDim

    B = B_N // numSubpatches

    # Reshape back to original dimensions
    x = x.view(B, numSubpatchesPerDim, numSubpatchesPerDim, numSubpatchesPerDim, C, P, P, P)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)  # [B, C, X//P, P, Y//P, P, Z//P, P]
    x = x.reshape(B, C, X, Y, Z)  # [B, C, X, Y, Z]

    return x

class DeepPatchEmbed3D(nn.Module):
    def __init__(self, channels: list[int], strides: list[int]):
        super().__init__()
        channels.insert(0, 1)
        self.channels = channels
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])

        for i in range(len(channels) - 1):
            block = nn.Sequential(
                nn.BatchNorm3d(channels[i]),
                nn.Conv3d(channels[i], channels[i], kernel_size=3, padding=1) if i == 0 else nn.Identity(),
                nn.Conv3d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm3d(channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=0.2) if i == len(channels) - 2 else nn.Identity(),
                nn.MaxPool3d(kernel_size=3, stride=strides[i], padding=1)
            )
            self.encoder.append(block)

    def forward(self, x: torch.Tensor, subPatchSize: int):  # x: [B, T, X, Y, Z]
        x, nSubPatches = subpatchTensor(x, subPatchSize)
        B, T, N, C, D, H, W = x.shape
        # N is num_patches
        # T is num_phases
        # C should be 1 since we want this to be our "channels" to become embed_dim
        x = x.reshape(B * T * N, C, D, H, W)
        # print("\tReshaped x grad_fn:", x.grad_fn)
        skips: list[torch.Tensor] = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            # print("\tAfter block x grad_fn:", x.grad_fn)
            unpatched = unpatchTensor(x, nSubPatches)
            # print(f"Skip {i} has shape: {unpatched.shape}")
            _, E, X, Y, Z = unpatched.shape
            unpatched = unpatched.reshape(B, T, E, X, Y, Z)
            unpatched = unpatched.mean(dim=1)       # for skip connections, reduce phases with mean
            skips.append(unpatched)

        _, E, X, Y, Z = x.shape
        # assert X == Y == Z == 1, f"Expected spatial dims to reduce to 1, got {X} x {Y} x {Z}"
        x = x.reshape(B, T, N, E, X, Y, Z)
        x = x.permute(0, 2, 4, 5, 6, 1, 3)      # [B, N, X, Y, Z, T, E] -> put N, X, Y, Z next to each other so they can be squished
        x = x.reshape(B, -1, T, E)              # [B, N*X*Y*Z, T, E]

        return x, skips, (B, T, N, E, X, Y, Z)

class ClassifierHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor):
        x = x.flatten(1)
        return self.fc(x) # (B, 1)

class TransformerST(nn.Module):
    def __init__(self,
                 patch_size,
                 channels,
                 strides, 
                 num_phases,
                 transformer_num_heads=4,
                 transformer_num_layers=6,
                 p_split=4):
        super().__init__()

        self.patch_size = patch_size
        self.p_split = p_split

        num_patches = round(self.p_split**3)
        self.num_patches = num_patches
        self.num_phases = num_phases

        self.emb_dim = channels[-1]

        nPatientDataInFeatures = 7             # 1 for age (linear), 2 for menopausal status (one-hot), 4 for breast density (one-hot)
        self.nPatientDataOutFeatures = transformer_num_heads         # needs to be divisible by num_heads
        self.patientDataEmbed = nn.Linear(nPatientDataInFeatures, self.nPatientDataOutFeatures)

        expectedXYZ = patch_size
        for _ in range(len(channels) + 1):
            expectedXYZ = max(round(expectedXYZ / 2), 1)

        self.patch_embed = DeepPatchEmbed3D(channels, strides)
        self.cls_token   = nn.Parameter(torch.randn((1, 1, num_phases, self.emb_dim + self.nPatientDataOutFeatures)))
        self.pos_embed   = nn.Parameter(torch.randn((1, num_patches * round(expectedXYZ**3) + 1, num_phases, self.emb_dim + self.nPatientDataOutFeatures)))
        
        ageMin = 21
        ageMax = 77
        self.ageEncode  = lambda x: torch.tensor([(x - ageMin) / (ageMax - ageMin)], dtype=torch.float32) if x is not None \
            else torch.tensor([0.5], dtype=torch.float32)   # normalize age to [0, 1] range, default to 0.5 if None
        self.menoEncode = lambda x: torch.tensor([1, 0]) if x == "pre" \
            else torch.tensor([0, 1]) if x == "post" \
            else torch.tensor([0, 0])
        self.densityEncode = lambda x: torch.tensor([1, 0, 0, 0]) if x == "a" \
            else torch.tensor([0, 1, 0, 0]) if x == "b" \
            else torch.tensor([0, 0, 1, 0]) if x == "c" \
            else torch.tensor([0, 0, 0, 1]) if x == "d" \
            else torch.tensor([0, 0, 0, 0])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim + self.nPatientDataOutFeatures,
            nhead=transformer_num_heads,
            dim_feedforward=(self.emb_dim + self.nPatientDataOutFeatures) * 4,
            batch_first=True
        )

        self.transformerT = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
        self.transformerS = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)

        self.fc_to_patches = nn.Linear(self.emb_dim + self.nPatientDataOutFeatures, self.emb_dim)
        self.temporal_proj = nn.Linear(self.num_phases, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize DeepPatchEmbed3D
        def init_weights_deep_patch_embed(module):
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        self.patch_embed.apply(init_weights_deep_patch_embed)

        # Initialize Transformer
        def init_weights_transformer(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        self.transformerT.apply(init_weights_transformer)
        self.transformerS.apply(init_weights_transformer)
    
    def forward(self, x: torch.Tensor, patientData: list[dict] = None):
        subpatchSize = self.patch_size // self.p_split
        
        x, skips, (B, T, N, E, X, Y, Z) = self.patch_embed(x, subpatchSize)  # [B, N*X*Y*Z, T, E]

        patientData_emb = torch.zeros(B, N*X*Y*Z, T, self.patientDataEmbed.out_features, device=x.device)      # [B, N, npatientDataOutFeatures]

        if patientData is None:
            patientData = [{'age': None, 'menopausal_status': None, 'breast_density': None} for _ in range(B)]

        # print(patientData)
        for idx, md in enumerate(patientData):
            age = md['age']
            menopausal_status = md['menopausal_status']
            breast_density = md['breast_density']
            age_emb = self.ageEncode(age).to(x.device)
            meno_emb = self.menoEncode(menopausal_status).to(x.device)
            density_emb = self.densityEncode(breast_density).to(x.device)
            concat = torch.cat((age_emb, meno_emb, density_emb), dim=0).unsqueeze(0)  # [1, npatientDataInFeatures]
            concat: torch.Tensor = self.patientDataEmbed(concat)  # [1, npatientDataOutFeatures]
            patientData_emb[idx] += concat.repeat(N*X*Y*Z, T, 1)  # [B, N*X*Y*Z, T, npatientDataOutFeatures]

        x = torch.cat((x, patientData_emb), dim=-1)  # [B, N*X*Y*Z, T, E + npatientDataOutFeatures]

        # prepend CLS token for classification prediction
        tok = self.cls_token.expand(B, -1, -1, -1)
        x = torch.cat((tok, x), dim=1)      # [B, N*X*Y*Z + 1, T, E + npatientDataOutFeatures]

        x = x + self.pos_embed

        # temporal transformer first
        x = self.transformerT(x.reshape(-1, T, E + self.nPatientDataOutFeatures))       # [B*(N*X*Y*Z + 1), T, E + npatientDataOutFeatures]

        x = x.reshape(B, N*X*Y*Z + 1, T, E + self.nPatientDataOutFeatures)
        x = x.permute(0, 1, 3, 2)           # [B, N*X*Y*Z + 1, E + npatientDataOutFeatures, T]
        x = self.temporal_proj(x)           # [B, N*X*Y*Z + 1, E + npatientDataOutFeatures, 1]
        x = x.squeeze(dim=-1)               # [B, N*X*Y*Z + 1, E + npatientDataOutFeatures]

        # then spatial transformer
        x = self.transformerS(x)            # [B, N*X*Y*Z + 1, E + npatientDataOutFeatures]
       
        x = self.fc_to_patches(x)  # [B, N*X*Y*Z + 1, E]
        cls_token = x[:, 0]   # [B, 1, E].

        # reshape to match conv shape
        x = x[:, 1:].reshape(B*N, X, Y, Z, E)
        x = x.permute(0, 4, 1, 2, 3)
        features = unpatchTensor(x, self.num_patches)       # [B, E, P, P, P]

        skips[-1] = features

        return features, skips, cls_token



if __name__ == "__main__":
    ...