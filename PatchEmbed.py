import torch
import torch.nn as nn

def init_weights_conv(module):
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.GroupNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

class PatchEncoder(nn.Module):
    def __init__(self, channels: list[int], strides: list[int], dropout: float = 0.2, useSkips: bool = False):
        super().__init__()
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        numBlocks = len(channels) - 1

        self.useSkips = useSkips

        # use the same encoder block design as my autoencoder! that worked pretty well!
        for i in range(numBlocks):
            encBlock = nn.Sequential(
                nn.Dropout3d(dropout) if i == numBlocks - 1 else nn.Identity(),
                nn.GroupNorm(num_groups=min(8, channels[i]), num_channels=channels[i]) if i != 0 else nn.Identity(),
                nn.Conv3d(channels[i], channels[i], kernel_size=3, padding=1) if i == 0 or i == 1 else nn.Identity(),
                nn.Conv3d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool3d(kernel_size=3, stride=strides[i], padding=1)
            )
            self.encoder.append(encBlock)

        self.init_weights()

    def forward(self, x: torch.Tensor):  # x: [..., X, Y, Z]
        x = x.unsqueeze(-4)     # add singleton channel dimension
        B, T, N, C, D, H, W = x.shape
        # N is num_patches
        # T is num_phases
        # C should be 1 since we want this to be our "channels" to become embed_dim
        x = x.reshape(B * T * N, C, D, H, W)

        skips: list[torch.Tensor] = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            if self.useSkips:
                unpatched = x.reshape(B, T, N, *x.shape[-4:])[:,1]          # reshape to [B, N, C, X, Y, Z], taking first post-contrast phase from T
                unpatched = unpatched.reshape(-1, *unpatched.shape[2:])     # merge the batch and patch dimensions to [B*N, C, X, Y, Z]
                skips.append(unpatched)        

        _, E, X, Y, Z = x.shape
        assert X == Y == Z == 1, f"Expected spatial dims to reduce to 1, got {X} x {Y} x {Z}"
        x = x.reshape(B, T, N, E, X, Y, Z)
        x = x.permute(0, 2, 4, 5, 6, 1, 3)      # [B, N, X, Y, Z, T, E] -> put N, X, Y, Z next to each other so they can be squished
        x = x.reshape(B, -1, T, E)              # [B, N*X*Y*Z, T, E]

        if self.useSkips:
            return x, skips, (B, T, N, E, X, Y, Z)
        else:
            return x, None, (B, T, N, E, X, Y, Z)

    def init_weights(self):
        self.apply(init_weights_conv)

class PatchDecoder(nn.Module):
    def __init__(self, channels: list[int], useSkips: bool = False):
        super().__init__()
        # print(f"DECODER CHANNELS: {channels}")     # [1, 32, 64, 128, 256, 320]
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        numBlocks = len(channels) - 1

        self.useSkips = useSkips
        for i in range(numBlocks):
            if self.useSkips:
                decBlock = nn.Sequential(
                    nn.ConvTranspose3d(2*channels[-(i+1)], channels[-(i+2)], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.Conv3d(channels[-(i+2)], channels[-(i+2)], kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=min(8, channels[-(i+2)]), num_channels=channels[-(i+2)]) if i < numBlocks - 1 else nn.Identity(),
                    nn.ReLU(True) if i < numBlocks - 1 else nn.Identity()
                )
            else:
                decBlock = nn.Sequential(
                    nn.ConvTranspose3d(channels[-(i+1)], channels[-(i+2)], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.Conv3d(channels[-(i+2)], channels[-(i+2)], kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=min(8, channels[-(i+2)]), num_channels=channels[-(i+2)]) if i < numBlocks - 1 else nn.Identity(),
                    nn.ReLU(True) if i < numBlocks - 1 else nn.Identity()
                )

            self.decoder.append(decBlock)
        
        self.init_weights()

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor] = None):
        # Decoder with skip connections
        if self.useSkips and skips:
            for i, block in enumerate(self.decoder):
                skip = skips[-(i + 1)]                      # Get corresponding skip connection
                # print(f"\tDoing skips with x: {x.shape}, skip: {skip.shape}")
                x = torch.cat((x, skip), dim=-4)            # Concatenate along channel dimension
                x = block(x)
        else:
            for block in self.decoder:
                x = block(x)

        return x    # output raw logits
    
    def init_weights(self):
        self.apply(init_weights_conv)