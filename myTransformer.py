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

    return x

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
    def __init__(self, channels: list[int], inChannels: int, strides: list[int]):
        super().__init__()
        channels.insert(0, inChannels)
        self.channels = channels
        self.encoder = nn.ModuleList([])
        
        for i in range(len(channels) - 1):
            block = nn.Sequential(
                nn.BatchNorm3d(num_features=channels[i]),
                nn.Conv3d(channels[i], channels[i], kernel_size=3, padding=1) if i == 0 else nn.Identity(),
                nn.Conv3d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm3d(num_features=channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout3d(0.3) if i == len(channels) - 2 else nn.Identity(),
                nn.MaxPool3d(kernel_size=3, stride=strides[i], padding=1)
            )
            self.encoder.append(block)

    def forward(self, x: torch.Tensor):  # x: [B, N, C, D, H, W]
        B, N, C, D, H, W = x.shape
        x = x.reshape(B * N, C, D, H, W)
        # print("\tReshaped x grad_fn:", x.grad_fn)
        for layer in self.encoder:
            x = layer(x)

        _, E, X, Y, Z = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(B, N*X*Y*Z, E)        # each token represents one small patch region with an E-dim embedding

        return x, (B, N, E, X, Y, Z)

class ClassifierHead(nn.Module):
    def __init__(self, dim, metadata_d):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim + metadata_d, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor):
        x = x.flatten(1)
        return self.fc(x) # (B, 1)

class MyTransformer(nn.Module):
    def __init__(
        self,
        channels,
        strides,
        in_channels,
        p_split,
        transformer_depth,
        num_heads,
    ):
        super().__init__()

        self.channels = channels
        self.inChannels = in_channels
        self.emb_dim = channels[-1]
        self.p_split = p_split

        self.patch_embed = DeepPatchEmbed3D(channels, in_channels, strides)

        nMetadataInFeatures = 7             # 1 for age (linear), 2 for menopausal status (one-hot), 4 for breast density (one-hot)
        nMetadataOutFeatures = num_heads    # needs to be divisible by num_heads
        self.metadataEmbed = nn.Linear(nMetadataInFeatures, nMetadataOutFeatures)

        self.cls_token = nn.Parameter(torch.randn((1, 1, self.emb_dim + nMetadataOutFeatures)))
        self.pos_embed = nn.Parameter(torch.randn((self.emb_dim + nMetadataOutFeatures,)))

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
            batch_first=True,
            dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

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
        subpatchSize = P // self.p_split       # split each patch into p_split**3 subpatches
        # print("Input x grad:", x.requires_grad)
        x = subpatchTensor(x, subpatchSize)
        # print("X shape after patching", x.shape)
        # print("Subpatched x grad:", x.grad_fn, " - nSubPatches:", nSubPatches)
        x, (B, N, E, X, Y, Z) = self.patch_embed(x)  # [B, N*X*Y*Z, emb_dim]
        # print(B, N, E, X, Y, Z)
        # print("patch embedded x grad_fn:", x.grad_fn)
        # for i, s in enumerate(skips):
        #     print(f"Skip {i} grad_fn: {s.grad_fn}")

        # print("Embedded x shape:", x.shape)

        metadata_emb = torch.zeros(B, N*X*Y*Z, self.metadataEmbed.out_features, device=x.device)  # [B, N*X*Y*Z, nMetadataOutFeatures]

        if metadata is None:
            metadata = [{'age': None, 'menopausal_status': None, 'breast_density': None} for _ in range(B)]

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
            metadata_emb[idx] += concat.repeat(N*X*Y*Z, 1)  # [B, N*X*Y*Z, nMetadataOutFeatures]

        # print("metadata shape", metadata_emb.shape)

        x = torch.cat((x, metadata_emb), dim=-1)  # [B, N, emb_dim + nMetadataOutFeatures]

        # prepend CLS token for classification prediction
        tok = self.cls_token.expand(B, -1, -1)
        # print("cls_tok expanded shape:", tok.shape)
        x = torch.cat((tok, x), dim=1)      # [B, N + 1, emb_dim + nMetadataOutFeatures]

        x = x + self.pos_embed

        x = self.transformer(x)

        # print("X shape after transformer pass:", x.shape)

        cls_out = x[:, 0]

        # print('cls_out.shape = ', cls_out.shape)
        # print("Final x grad_fn:", x.grad_fn)

        # print("Final x shape:", x.shape)
        # print("Final skips shape:", [s.shape for s in skips])
        # print("Final tokens shape:", tokens.shape)
        # print("x == last layer?", torch.all(x == tf_skips[-1]))

        return cls_out

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch = 2
    p = 128
    b = 2
    # imgname = r"./allTheData/HeatmapsAugmented/Training/00014_Image0.zarr"
    t = torch.rand((b, ch, p, p, p))
    subpatches, n = subpatchTensor(t, 32)
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