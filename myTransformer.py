import torch
import torch.nn as nn

def subpatchTensor(x: torch.Tensor, subpatchSize: int):
    assert len(x.shape) == 5, f"Expected shape (B, C, X, Y, Z). Got: {x.shape}"
    B, C, X, Y, Z = x.shape
    assert X % subpatchSize == 0 and Y % subpatchSize == 0 and Z % subpatchSize == 0, \
        "Input dimensions must be divisible by subpatchSize"

    # Reshape into subpatches
    x = x.unfold(2, subpatchSize, subpatchSize)  # Unfold X
    x = x.unfold(3, subpatchSize, subpatchSize)  # Unfold Y
    x = x.unfold(4, subpatchSize, subpatchSize)  # Unfold Z
    x = x.permute(0, 2, 3, 4, 1, 5, 6, 7)  # [B, X//subpatchSize, Y//subpatchSize, Z//subpatchSize, C, subpatchSize, subpatchSize, subpatchSize]
    x = x.reshape(B, -1, C, subpatchSize, subpatchSize, subpatchSize)  # [B, N, C, subpatchSize, subpatchSize, subpatchSize]

    numSubpatches = (X // subpatchSize) * (Y // subpatchSize) * (Z // subpatchSize)
    return x, numSubpatches

# this function undoes subpatching, returning a tensor to its original size
# (B*N, C, P, P, P) -> (B, C, X, Y, Z)
# X == Y == Z
# N == numSubpatches
def unpatchTensor(x: torch.Tensor, numSubpatches: int):
    assert len(x.shape) == 5, f"Expected shape (B*N, C, P, P, P). Got {x.shape}"
    B_N, C, P, _, _ = x.shape
    numSubpatchesPerDim = int(numSubpatches ** (1/3))
    X = Y = Z = P * numSubpatchesPerDim

    B = B_N // numSubpatches

    # Reshape back to original dimensions
    x = x.view(B, numSubpatchesPerDim, numSubpatchesPerDim, numSubpatchesPerDim, C, P, P, P)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)  # [B, C, X//P, P, Y//P, P, Z//P, P]
    x = x.reshape(B, C, X, Y, Z)  # [B, C, X, Y, Z]

    return x

class DeepPatchEmbed3D(nn.Module):
    def __init__(self, channels: list[int], inChannels: int, strides: list[int]):
        super().__init__()
        encoder = nn.ModuleList([])
        channels.insert(0, inChannels)
        self.channels = channels

        groups = [max(min(8, ch // 8), 1) for ch in channels]
        
        for i in range(len(channels) - 1):
            block = nn.ModuleList([])
            block.append(nn.GroupNorm(num_groups=groups[i], num_channels=channels[i]))
            if i == 0:
                block.append(nn.Conv3d(channels[i], channels[i], kernel_size=3, padding=1))
            block.append(nn.Conv3d(channels[i], channels[i + 1], kernel_size=3, padding=1))   # these convs don't change output size because of padding
            block.append(nn.ReLU(inplace=True))
            block.append(nn.MaxPool3d(kernel_size=3, stride=strides[i], padding=1))    # this stride changes output size

            encoder.append(nn.Sequential(*block))
        
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x: torch.Tensor, nSubPatches: int):  # x: [B, N, C, D, H, W]
        B, N, C, D, H, W = x.shape
        x = x.reshape(B * N, C, D, H, W)
        # print("\tReshaped x grad_fn:", x.grad_fn)
        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            # print("\tAfter block x grad_fn:", x.grad_fn)
            unpatched = unpatchTensor(x, nSubPatches)
            skips.append(unpatched)

        _, E, X, Y, Z = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(B, N*X*Y*Z, E)

        return x, skips, (B, N, E, X, Y, Z)

class MyTransformer(nn.Module):
    def __init__(
        self,
        channels,
        strides,
        in_channels=2,
        transformer_depth=4,
        num_heads=8,
    ):
        super().__init__()

        self.channels = channels
        self.inChannels = in_channels
        self.emb_dim = channels[-1]

        print(f"channels: {self.channels}")
        print(f"inChannels: {self.inChannels}")

        self.patch_embed = DeepPatchEmbed3D(channels, in_channels, strides)
        self.pos_embed = nn.Parameter(torch.randn((self.emb_dim,)), requires_grad=True)  # learnable position embedding
        
        nMetadataInFeatures = 7             # 1 for age (linear), 2 for menopausal status (one-hot), 4 for breast density (one-hot)
        nMetadataOutFeatures = num_heads    # needs to be divisible by num_heads
        self.metadataEmbed = nn.Linear(nMetadataInFeatures, nMetadataOutFeatures)

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
            d_model=self.emb_dim + nMetadataOutFeatures,
            nhead=num_heads,
            dim_feedforward=(self.emb_dim + nMetadataOutFeatures) * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

        self.fc_to_patches = nn.Linear(self.emb_dim + nMetadataOutFeatures, self.emb_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize DeepPatchEmbed3D
        def init_weights_deep_patch_embed(module):
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
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

        self.transformer.apply(init_weights_transformer)

    def forward(self, x: torch.Tensor, metadata: list = None):
        # print("X dist:", x.mean(), "+/-", x.std())
        P = x.shape[-1]
        # Embed patches
        subpatchSize = P // 2       # split each patch into 8 subpatches
        # print("Input x grad:", x.requires_grad)
        x, nSubPatches = subpatchTensor(x, subpatchSize)
        # print("Subpatched x grad:", x.grad_fn, " - nSubPatches:", nSubPatches)
        x, skips, (B, N, E, X, Y, Z) = self.patch_embed(x, nSubPatches)  # [B, N, emb_dim]
        # print("patch embedded x grad_fn:", x.grad_fn)
        # for i, s in enumerate(skips):
        #     print(f"Skip {i} grad_fn: {s.grad_fn}")
        x = x + self.pos_embed

        # print("Embedded x shape:", x.shape)

        metadata_emb = torch.zeros(B, N, self.metadataEmbed.out_features, device=x.device)  # [B, N, nMetadataOutFeatures]
        # print(metadata)
        for idx, md in enumerate(metadata):
            age = md['age']
            menopausal_status = md['menopausal_status']
            breast_density = md['breast_density']
            age_emb = self.ageEncode(age).to(x.device)
            meno_emb = self.menoEncode(menopausal_status).to(x.device)
            density_emb = self.densityEncode(breast_density).to(x.device)
            concat = torch.cat((age_emb, meno_emb, density_emb), dim=0).unsqueeze(0)  # [1, nMetadataInFeatures]
            concat: torch.Tensor = self.metadataEmbed(concat)  # [1, nMetadataOutFeatures]
            metadata_emb[idx] += concat.repeat(N, 1)  # [B, N, nMetadataOutFeatures]

        x = torch.cat((x, metadata_emb), dim=-1)  # [B, N, emb_dim + nMetadataOutFeatures]

        x: torch.Tensor = self.transformer(x)  # [B, N, emb_dim + nMetadataOutFeatures]
        # print("Transformer raw output grad_fn:", x.grad_fn)
        x = self.fc_to_patches(x)  # [B, N, emb_dim]

        # reshape to match conv shape
        x = x.reshape(B*N, X, Y, Z, E)
        x = x.permute(0, 4, 1, 2, 3)
        x = unpatchTensor(x, nSubPatches)
        # print("Final x grad_fn:", x.grad_fn)

        return x, skips

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch = 2
    p = 128
    b = 2
    # imgname = r"./allTheData/HeatmapsAugmented/Training/00014_Image0.zarr"
    t = torch.rand((b, ch, p, p, p))
    subpatches, n = subpatchTensor(t.clone(), 64)
    print(subpatches.shape)
    x, y, z = subpatches.shape[-3:]
    unpatched = unpatchTensor(subpatches.view(b*n, ch, x, y, z), n)
    print(unpatched.shape)
    print(torch.all(t == unpatched))
    # t = CrossPatchTransformerAE3D(device=device,
    #                               in_channels=ch,
    #                               out_channels=ch,
    #                               num_patches=n).to(device)
    # testInput = torch.rand(b, n, ch, p, p, p).round().float().to(device)
    # print(testInput.min(), testInput.mean(), testInput.max())
    # output: torch.Tensor = t(testInput)
    # print(output.shape)
    # print(output.min(), output.mean(dtype=torch.float16), output.max())