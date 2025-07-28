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

        self.p_split = p_split

        num_patches = round(self.p_split**3)
        self.num_patches = num_patches
        self.num_phases = num_phases

        self.emb_dim = channels[-1]

        expectedXYZ = patch_size
        for _ in range(len(channels) + 1):
            expectedXYZ = max(round(expectedXYZ / 2), 1)

        self.patch_embed = DeepPatchEmbed3D(channels, strides)
        
        self.pos_embed      = nn.Parameter(torch.randn((1, num_patches * round(expectedXYZ**3), num_phases, self.emb_dim)))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=transformer_num_heads,
            dim_feedforward=(self.emb_dim) * 4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)

        self.temporal_proj = nn.Linear(self.num_phases, 1)

    def forward(self, x: torch.Tensor):
        # print(x.shape)      # [B, T, X, Y, Z]

        P = x.shape[-1]
        # Embed patches
        subpatchSize = P // self.p_split
        
        x, skips, (B, T, N, E, X, Y, Z) = self.patch_embed(x, subpatchSize)  # [B, N*X*Y*Z, T, E]

        x = x + self.pos_embed

        # print(f"embedded x shape: {x.shape}")

        # temporal transformer first
        x = self.transformer(x.reshape(-1, T, E))       # [B*N*X*Y*Z, T, E]

        x = x.reshape(B, N*X*Y*Z, T, E)
        x = x.permute(0, 1, 3, 2)           # [B, N*X*Y*Z, E, T]
        x = self.temporal_proj(x)           # [B, N*X*Y*Z, E, 1]
        x = x.squeeze(dim=-1)                     # [B, N*X*Y*Z, E]
        # print(f"temporally projected x: {x.shape}")

        # then spatial transformer
        x = self.transformer(x)             # [B, N*X*Y*Z, E]

        # print(f"x shape after transformer: {x.shape}")

        tf_skip = x.permute(0, 2, 1)        # [B, E, N*X*Y*Z]
        tf_skip = tf_skip.reshape(B, E, self.p_split*X, self.p_split*Y, self.p_split*Z)
        # print("tf_skip size", tf_skip.shape)

        # reshape to match conv shape
        x = x.reshape(B*N, X, Y, Z, E)
        x = x.permute(0, 4, 1, 2, 3)
        features = unpatchTensor(x, self.num_patches)       # [B, E, P, P, P]
        # print(f"Unpatched features shape: {features.shape}")

        # print("x shape before reshape:", x.shape)
        # print("Final x grad_fn:", x.grad_fn)

        # print("Final x shape:", x.shape)
        # print("Final skips shape:", [s.shape for s in skips])
        # print("Final tokens shape:", tokens.shape)
        # print("x == last layer?", torch.all(x == tf_skips[-1]))

        return features, skips, None



if __name__ == "__main__":
    ...