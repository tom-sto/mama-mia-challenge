import torch
import torch.nn as nn

def subpatchTensor(x: torch.Tensor, subpatchSize: int):
    assert len(x.shape) == 5, f"Expected shape (B, C, X, Y, Z). Got: {x.shape}"
    patches = []
    tensorShape = torch.tensor(x.shape)
    patchSize = tensorShape[-3:]
    indexShape = (tensorShape / subpatchSize).int()
    numSubpatches = (patchSize / subpatchSize).prod().int().item()
    for i in range(numSubpatches):
        w_i = i % indexShape[-1]
        h_i = (i // indexShape[-1]) % indexShape[-2]
        d_i = (i // (indexShape[-1] * indexShape[-2])) % indexShape[-3]
        patch = x[:, :,
            d_i * subpatchSize:(d_i + 1) * subpatchSize,
            h_i * subpatchSize:(h_i + 1) * subpatchSize,
            w_i * subpatchSize:(w_i + 1) * subpatchSize
        ]
        # print(f"patch shape: {patch.shape}")
        patches.append(patch)

    subpatched = torch.stack(patches).permute(1, 0, 2, 3, 4, 5)
    return subpatched, numSubpatches

# this function undoes subpatching, returning a tensor to its original size
# (B*N, C, P, P, P) -> (B, C, X, Y, Z)
# X == Y == Z
# N == numSubpatches
def unpatchTensor(x: torch.Tensor, numSubpatches: int):
    assert len(x.shape) == 5, f"Expected shape (B*N, C, P, P, P). Got {x.shape}"
    B_N, C, P, _, _ = x.shape
    B = B_N // numSubpatches
    subpatchSize = P
    numSubpatchesPerDim = int(numSubpatches ** (1/3))
    X = Y = Z = subpatchSize * numSubpatchesPerDim

    # Initialize the output tensor
    output = torch.zeros((B, C, X, Y, Z), device=x.device, dtype=x.dtype)

    for i in range(numSubpatches):
        w_i = i % numSubpatchesPerDim
        h_i = (i // numSubpatchesPerDim) % numSubpatchesPerDim
        d_i = (i // (numSubpatchesPerDim ** 2)) % numSubpatchesPerDim

        patch = x[i::numSubpatches]  # Select patches for each batch
        output[:, :, 
               d_i * subpatchSize:(d_i + 1) * subpatchSize,
               h_i * subpatchSize:(h_i + 1) * subpatchSize,
               w_i * subpatchSize:(w_i + 1) * subpatchSize] = patch

    return output

class DeepPatchEmbed3D(nn.Module):
    def __init__(self, channels: list[int], inChannels: int, strides: list[int]):
        super().__init__()
        encoder = nn.ModuleList([])
        channels.insert(0, inChannels)
        print("Channels:", channels)

        groups = [max(min(8, ch // 4), 1) for ch in channels]
        print("Groups:", groups)
        
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
        print("encoder in shape", x.shape)
        # x = self.encoder(x)              # [B*N, emb_dim, ...]
        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            print(f"Shape after layer {i}: {x.shape}")
            unpatched = unpatchTensor(x, nSubPatches)
            skips.append(unpatched)
        print("encoder out shape", x.shape)
        return x, skips

class MyTransformer(nn.Module):
    def __init__(
        self,
        channels,
        strides,
        in_channels=2,
        transformer_depth=2,
        num_heads=4,
    ):
        super().__init__()

        self.channels = channels
        self.inChannels = in_channels
        self.emb_dim = channels[-1]

        self.patch_embed = DeepPatchEmbed3D(channels, in_channels, strides)

        self.pos_embed = nn.Parameter(torch.randn((self.emb_dim,), requires_grad=True))  # learnable position embedding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=num_heads,
            dim_feedforward=self.emb_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

    def forward(self, x: torch.Tensor):
        # Embed patches
        #TODO: Sub-patching here!
        subpatchSize = 64
        x, nSubPatches = subpatchTensor(x, subpatchSize)
        print("subpatch shape:", x.shape)
        x, skips = self.patch_embed(x, nSubPatches)  # [B, N, emb_dim]
        print("patch embed shape:", x.shape)
        print("Skips:", skips)
        # Add positional encoding
        x = x + self.pos_embed

        # Transformer encoder
        x: torch.Tensor = self.transformer(x)  # [B, N, emb_dim]
        print("Transformer shape:", x.shape)
        return x

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch = 2
    p = 128
    b = 2
    imgname = r"./allTheData/HeatmapsAugmented/Training/00014_Image0.zarr"
    x = torch.rand((b, ch, p, p, p))
    subpatches = subpatchTensor(x)
    print(subpatches.shape)
    # t = CrossPatchTransformerAE3D(device=device,
    #                               in_channels=ch,
    #                               out_channels=ch,
    #                               num_patches=n).to(device)
    # testInput = torch.rand(b, n, ch, p, p, p).round().float().to(device)
    # print(testInput.min(), testInput.mean(), testInput.max())
    # output: torch.Tensor = t(testInput)
    # print(output.shape)
    # print(output.min(), output.mean(dtype=torch.float16), output.max())