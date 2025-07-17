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

        # Create decoder blocks (mirrored but reinitialized)
        for i in reversed(range(len(channels) - 1)):
            block = nn.Sequential(
                nn.BatchNorm3d(channels[i + 1]),
                nn.Conv3d(channels[i + 1], channels[i], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(channels[i], channels[i], kernel_size=3, stride=strides[i], padding=1, output_padding=1)
            )
            self.decoder.append(block)


    def forward(self, x: torch.Tensor, nSubPatches: int):  # x: [B, N, C, D, H, W]
        B, N, C, D, H, W = x.shape
        x = x.reshape(B * N, C, D, H, W)
        # print("\tReshaped x grad_fn:", x.grad_fn)
        skips: list[torch.Tensor] = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            # print("\tAfter block x grad_fn:", x.grad_fn)
            unpatched = unpatchTensor(x, nSubPatches)
            skips.append(unpatched)

        _, E, X, Y, Z = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(B, N*X*Y*Z, E)

        return x, skips, (B, N, E, X, Y, Z)

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

class MyTransformer(nn.Module):
    def __init__(
        self,
        patch_size,
        channels,
        strides,
        in_channels,
        transformer_depth=6,    # make this match num layers in decoder for skips: 6
        num_heads=10,
        p_split=4
    ):
        super().__init__()

        self.channels = channels
        self.inChannels = in_channels
        self.emb_dim = channels[-1]
        self.p_split = p_split

        nPatientDataInFeatures = 7             # 1 for age (linear), 2 for menopausal status (one-hot), 4 for breast density (one-hot)
        nPatientDataOutFeatures = num_heads    # needs to be divisible by num_heads
        self.patientDataEmbed = nn.Linear(nPatientDataInFeatures, nPatientDataOutFeatures)

        expectedXYZ = patch_size
        for _ in range(len(channels) + 1):
            expectedXYZ = max(round(expectedXYZ / 2), 1)

        self.patch_embed = DeepPatchEmbed3D(channels, in_channels, strides)
        self.cls_token = nn.Parameter(torch.randn((1, 1, self.emb_dim + nPatientDataOutFeatures)))
        self.pos_embed = nn.Parameter(torch.randn((1, 
                                                   round(p_split**3) * round(expectedXYZ**3) + 1, 
                                                   self.emb_dim + nPatientDataOutFeatures,)))  # learnable position embedding
        self.skip_weights = nn.Parameter(torch.ones(len(self.patch_embed.decoder)) * 0.5)  # learnable skip connection weights

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
            d_model=self.emb_dim + nPatientDataOutFeatures,
            nhead=num_heads,
            dim_feedforward=(self.emb_dim + nPatientDataOutFeatures) * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

        self.fc_to_patches = nn.Linear(self.emb_dim + nPatientDataOutFeatures, self.emb_dim)

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

        self.transformer.apply(init_weights_transformer)

    def forward(self, x: torch.Tensor, patientData: list = None):
        # print("X dist:", x.mean(), "+/-", x.std())
        P = x.shape[-1]
        # Embed patches
        subpatchSize = P // self.p_split       # split each patch into p_split**3 subpatches
        # print("Input x grad:", x.requires_grad)
        x, nSubPatches = subpatchTensor(x, subpatchSize)
        # print("Subpatched x grad:", x.grad_fn, " - nSubPatches:", nSubPatches)
        x, skips, (B, N, E, X, Y, Z) = self.patch_embed(x, nSubPatches)  # [B, N, emb_dim]
        # print(B, N, E, X, Y, Z)
        # X, Y, Z should be 1 if the dimensions align well and there are enough conv layers. 
        # But they might not get down to 1, so we use them here to get the right size embedding
        patientData_emb = torch.zeros(B, N*X*Y*Z, self.patientDataEmbed.out_features, device=x.device)  # [B, N, npatientDataOutFeatures]

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
            patientData_emb[idx] += concat.repeat(N*X*Y*Z, 1)  # [B, N*X*Y*Z, npatientDataOutFeatures]

        x = torch.cat((x, patientData_emb), dim=-1)  # [B, N*X*Y*Z, emb_dim + npatientDataOutFeatures]

        # prepend CLS token for classification prediction
        tok = self.cls_token.expand(B, -1, -1)
        x = torch.cat((tok, x), dim=1)      # [B, N*X*Y*Z + 1, emb_dim + npatientDataOutFeatures]
        x = x + self.pos_embed

        tf_skips: list[torch.Tensor] = []
        for layer in self.transformer.layers:
            # print("Layer input grad_fn:", x.grad_fn)
            x = layer(x)

            y: torch.Tensor = self.fc_to_patches(x)[:, 1:]  # [B, N*X*Y*Z, emb_dim]
            y = y.permute(0, 2, 1)  # [B, emb_dim, N*X*Y*Z]
            y = y.reshape(B, E, self.p_split*X, self.p_split*Y, self.p_split*Z)       # [B, emb_dim, p_split, p_split, p_split]
            tf_skips.append(y)

        x = self.fc_to_patches(x)  # [B, N*X*Y*Z + 1, emb_dim]
        tokens = x.clone()[:, 0]   # [B, 1, emb_dim]

        tf_skips = tf_skips[::-1]
        for i in range(len(tf_skips)):
            expected_shape = skips[-(i+1)].shape
            if i == 0:
                assert tf_skips[i].shape == expected_shape, f"Skip {i} shape {tf_skips[i].shape} does not match expected shape {expected_shape}"
                continue
            for j in range(i):
                tf_skips[i] = self.patch_embed.decoder[j](tf_skips[i])
            assert tf_skips[i].shape == expected_shape, f"Skip {i} shape {tf_skips[i].shape} does not match expected shape {expected_shape}"

        tf_skips = tf_skips[::-1]

        for i in range(len(skips)):
            skips[i] = (1 - self.skip_weights[i]) * skips[i] + self.skip_weights[i] * tf_skips[i]

        # print("x shape before reshape:", x.shape)

        # reshape to match conv shape
        x = x[:, 1:].reshape(B*N, X, Y, Z, E)
        x = x.permute(0, 4, 1, 2, 3)
        x = unpatchTensor(x, nSubPatches)
        # print("Final x grad_fn:", x.grad_fn)

        # print("Final x shape:", x.shape)
        # print("Final skips shape:", [s.shape for s in skips])
        # print("Final tokens shape:", tokens.shape)
        # print("x == last layer?", torch.all(x == tf_skips[-1]))

        return x, skips, tokens

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