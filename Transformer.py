import torch
import torch.nn as nn

def subpatchTensor(x: torch.Tensor, subpatchSize: int = 64):
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
        patches.append(patch.unsqueeze(0))

    subpatched = torch.cat(patches).permute(1, 0, 2, 3, 4, 5)
    return subpatched



class DeepPatchEmbed3D(nn.Module):
    def __init__(self, in_channels=6, emb_dim=128, num_layers=3):
        super().__init__()
        encoder = nn.ModuleList([])
        start = torch.log2(torch.tensor(in_channels))
        end = torch.log2(torch.tensor(emb_dim))
        channels = torch.logspace(start, end, num_layers + 1, base=2).int().tolist()

        for i in range(num_layers):
            encoder.append(nn.GroupNorm(num_groups=1, num_channels=channels[i]))
            encoder.append(nn.Conv3d(channels[i], channels[i + 1], kernel_size=3, padding=1))
            encoder.append(nn.ReLU(inplace=True))
            encoder.append(nn.MaxPool3d(kernel_size=3, stride=2, padding=1))

        self.encoder = nn.Sequential(*encoder)

    def forward(self, x: torch.Tensor):  # x: [B, N, C, D, H, W]
        B, N, C, D, H, W = x.shape
        x = x.view(B * N, C, D, H, W)
        print("encoder in shape", x.shape)
        x = self.encoder(x)              # [B*N, emb_dim, ...]
        print("encoder out shape", x.shape)
        x = x.mean(dim=[-3, -2, -1], keepdim=True) # Global average pooling
        x = x.view(B, N, -1)             # [B, N, emb_dim]
        return x

class DeepDecoder3D(nn.Module):
    def __init__(self, emb_dim=128, out_channels=6, out_size=(16, 16, 16), num_layers=3):
        super().__init__()
        decoder = nn.ModuleList([])
        channels = torch.linspace(emb_dim, out_channels, num_layers + 1).int().tolist()

        for i in range(num_layers):
            decoder.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))
            decoder.append(nn.Conv3d(channels[i], channels[i+1], kernel_size=3, padding=1))
            if i == num_layers - 1:
                decoder.append(nn.AdaptiveAvgPool3d(out_size))  # Final output shape
            else:
                decoder.append(nn.BatchNorm3d(channels[i + 1]))
                decoder.append(nn.ReLU(inplace=True))

        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):  # x: [B, N, emb_dim]
        B, N, D = x.shape
        x = x.view(B * N, D, 1, 1, 1)
        x = self.decoder(x)  # [B*N, C, D', H', W']
        x = x.view(B, N, -1, *x.shape[2:])  # [B, N, out_channels, D', H', W']
        return x

class Encoder3D(nn.Module):
    def __init__(
        self,
        in_channels=2,
        emb_dim=128,
        num_patches=8,
        patch_size=128,
        num_layers=4,
        transformer_depth=2,
        num_heads=4,
    ):
        super().__init__()

        self.numPatches = num_patches   # num patches per big patch
        self.patchSize = patch_size     # side length of patch cube
        self.inChannels = in_channels

        self.patch_embed = DeepPatchEmbed3D(in_channels, emb_dim, num_layers)

        self.pos_embed = nn.Parameter(torch.randn((emb_dim,), requires_grad=True))  # learnable position embedding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

    def forward(self, x: torch.Tensor):
        # Embed patches
        #TODO: Sub-patching here!
        x = subpatchTensor(x)
        x = self.patch_embed(x)  # [B, N, emb_dim]

        # Add positional encoding
        x = x + self.pos_embed

        # Transformer encoder
        x: torch.Tensor = self.transformer(x)  # [B, N, emb_dim]
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