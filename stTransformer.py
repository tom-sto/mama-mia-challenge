import torch
import torch.nn as nn

def st_parallel():
    n_heads = 4
    emb_dim = 32
    patient_data_features = 8
    d_model = emb_dim + patient_data_features
    n_patches = 8
    batch_size = 3
    d_k = d_v = d_model // n_heads
    assert d_model % n_heads == 0, "Need d_model to be divisible by the number of heads"

    ST = 2

    # simulate embedding matrix after encoding, patient data embedding, and CLS token
    E = torch.randn(ST, batch_size, n_patches + 1, d_model)
    # simulate 5 phases (these will need to be encoded separately)
    E[1, :, 5:] = 0

    print(f"E: {E.shape}")

    W_q = torch.randn(ST, n_heads, d_model, d_k, requires_grad=True)
    W_k = torch.randn(ST, n_heads, d_model, d_k, requires_grad=True)
    W_v = torch.randn(ST, n_heads, d_model, d_v, requires_grad=True)
    W_o = torch.randn(ST, n_heads * d_v, d_model, requires_grad=True)

    Q = E @ W_q
    K = E @ W_k
    V = E @ W_v
    K_T = K.transpose(-1, -2)

    print(f"Q: {Q.shape}\t K: {K.shape}\t V: {V.shape}")
    print(f"K_T: {K_T.shape}")
    
    attn_pre_softmax = (K_T @ Q) / torch.sqrt(torch.tensor(d_k))
    print(f"Attn_pre_softmax: {attn_pre_softmax.shape}")
    
    attn = torch.softmax(attn_pre_softmax, 3)
    print(f"Attn: {attn.shape}")
    
    out = V @ attn
    print(f"out: {out.shape}")

# when running attention on one, assume the other is the "embedding" of that sequence
def st_sequential():
    n_heads = 2
    n_patches = 10
    n_phases = 6
    batch_size = 3

    d = [n_patches, n_phases]

    # simulate embedding matrix after encoding, patient data embedding
    E = torch.randn(batch_size, n_phases, n_patches)

    W_qs = torch.randn(n_heads, n_patches, n_patches // n_heads, requires_grad=True)
    W_ks = torch.randn(n_heads, n_patches, n_patches // n_heads, requires_grad=True)
    W_vs = torch.randn(n_heads, n_patches, n_patches // n_heads, requires_grad=True)
    W_os = torch.nn.Linear(n_patches, n_patches)

    W_qt = torch.randn(n_heads, n_phases, n_phases // n_heads, requires_grad=True)
    W_kt = torch.randn(n_heads, n_phases, n_phases // n_heads, requires_grad=True)
    W_vt = torch.randn(n_heads, n_phases, n_phases // n_heads, requires_grad=True)
    W_ot = torch.nn.Linear(n_phases, n_phases)

    W_q = [W_qs, W_qt]
    W_k = [W_ks, W_kt]
    W_v = [W_vs, W_vt]
    W_o = [W_os, W_ot]

    print(f"E: {E.shape}")

    for i in range(2):
        print(f"Step {i}")
        d_model = d[i]
        d_k = d_model // n_heads
        assert d_model % n_heads == 0, "Need d_model to be divisible by the number of heads"

        # Transpose E to align the axis of interest to last
        if i == 0:  # spatial attention
            x = E  # shape: (B, T, P)
        else:      # temporal attention
            x = E.transpose(1, 2)  # shape: (B, P, T)
        print(f"\tShape x: {x.shape}")

        B, T, D = x.shape  # T = sequence length (phases or patches), D = model dim

        # Project for all heads in parallel
        Q = torch.einsum("btd, hdk -> bhtk", x, W_q[i])
        K = torch.einsum("btd, hdk -> bhtk", x, W_k[i])
        V = torch.einsum("btd, hdk -> bhtk", x, W_v[i])

        print(f"\tQ shape: {Q.shape}")
        print(f"\tK shape: {K.shape}")
        print(f"\tV shape: {V.shape}")

        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        print(f"\tScores shape: {scores.shape}")

        attn = torch.softmax(scores, dim=-1)
        print(f"\tattn shape: {attn.shape}")

        context = torch.matmul(attn, V)  # (B, h, T, d_k)
        print(f"\tcontext shape: {context.shape}")

        # Concatenate heads
        context = context.permute(0, 2, 1, 3).contiguous()  # (B, T, h, d_k)
        context = context.view(B, T, -1)  # (B, T, h * d_k)
        print(f"\tcontext shape (after concat): {context.shape}")

        O = W_o[i](context)
        print(f"\tO shape = {O.shape}")

        # E = x + O
        E = O



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

        # Create decoder blocks (mirrored but reinitialized)
        for i in reversed(range(len(channels) - 1)):
            if i == 0: continue
            block = nn.Sequential(
                nn.BatchNorm3d(channels[i + 1]),
                nn.Conv3d(channels[i + 1], channels[i], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(channels[i], channels[i], kernel_size=3, stride=strides[i], padding=1, output_padding=1)
            )
            self.decoder.append(block)


    def forward(self, x: torch.Tensor, subPatchSize: int):  # x: [B, T, X, Y, Z]
        x, nSubPatches = subpatchTensor(x, subPatchSize)
        B, N, T, C, D, H, W = x.shape
        # N is num_patches
        # T is num_phases
        # C should be 1 since we want this to be our "channels" to become embed_dim
        x = x.reshape(B * N * T, C, D, H, W)
        # print("\tReshaped x grad_fn:", x.grad_fn)
        skips: list[torch.Tensor] = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            # print("\tAfter block x grad_fn:", x.grad_fn)
            unpatched = unpatchTensor(x, nSubPatches)
            skips.append(unpatched)

        _, E, X, Y, Z = x.shape
        assert X == Y == Z == 1, f"Expected spatial dims to reduce to 1, got {X} x {Y} x {Z}"
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(B, N, T, -1)

        return x, skips, (B, N, T, E, X, Y, Z)


class TransformerST(nn.Module):
    def __init__(self,
                    channels,
                    strides, 
                    num_phases,
                    patch_head_num,
                    phase_head_num,
                    attention_order = ['patch', 'phase'],
                    transformer_num_heads = 4,
                    transformer_num_layers = 6,
                    p_split=4):
        super().__init__()

        self.p_split = p_split

        num_patches = round(self.p_split**3)
        self.num_patches = num_patches
        self.num_phases = num_phases

        self.patch_head_num = patch_head_num
        self.phase_head_num = phase_head_num

        self.emb_dim = channels[-1]

        self.attention_layers = nn.ModuleList()
        self.attention_order = []
        for attn_type in attention_order:
            if attn_type == 'patch' and self.patch_head_num > 0:

                self.attention_layers.append(
                    nn.MultiheadAttention(
                        embed_dim=self.emb_dim,
                        num_heads=self.patch_head_num,
                        batch_first=True
                    )
                )
                self.attention_order.append('patch')
            if attn_type == 'phase' and self.phase_head_num > 0:
                self.attention_layers.append(
                    nn.MultiheadAttention(
                        embed_dim=self.emb_dim,
                        num_heads=self.phase_head_num,
                        batch_first=True
                    )
                )
                self.attention_order.append('phase')

        self.patch_embed = DeepPatchEmbed3D(channels, strides)
        
        self.pos_embed      = nn.Parameter(torch.randn((1, num_patches, num_phases, self.emb_dim)))
        self.skip_weights   = nn.Parameter(torch.randn((len(channels) - 2,)))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=transformer_num_heads,
            dim_feedforward=(self.emb_dim) * 4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
    

    def forward(self, x: torch.Tensor):
        print(x.shape)      # [B, T, X, Y, Z]

        P = x.shape[-1]
        # Embed patches
        subpatchSize = P // self.p_split
        
        x, skips, (B, N, T, E, X, Y, Z) = self.patch_embed(x, subpatchSize)  # [B, N, T, E]

        x = x + self.pos_embed

        print(f"embedded x shape: {x.shape}")

        # Attention
        patch_weights = None
        phase_weights = None
        for i, module in enumerate(self.attention_layers):
            attn_type = self.attention_order[i]
            
            if attn_type == 'patch':
                x = x.permute(0, 2, 1, 3)               # [B, T, N, E]
                x = x.reshape(-1, N, E)                 # [BT, N, E]
                x, patch_weights = module(x, x, x)
                x = x.reshape(B, T, N, E)
                x = x.permute(0, 2, 1, 3)               # [B, N, T, E]
            
            if attn_type == 'phase':
                x = x.reshape(-1, T, E)
                x, phase_weights = module(x, x, x)
                x = x.reshape(B, N, T, E)

        x = self.transformer(x.reshape(B, -1, E))       # [B, N*T, E]

        print(f"x shape after transformer: {x.shape}")

        tf_skip = x.reshape(B, N, T, E)
        tf_skip = tf_skip.permute(0, 2, 3, 1)  # [B, T, E, N]
        tf_skip = tf_skip.reshape(B*T, E, self.p_split, self.p_split, self.p_split)
        # print("tf_skip size", tf_skip.shape)

        # reshape to match conv shape
        x = x.reshape(B*N*T, 1, 1, 1, E)
        x = x.permute(0, 4, 1, 2, 3)
        features = unpatchTensor(x, self.num_patches)       # [B*T, E, P, P, P]
        print(f"Unpatched features shape: {features.shape}")

        tf_skips = [features]       # Last input here is the "bottleneck" features for the decoder
        for j in range(len(self.patch_embed.decoder) - 1):
            tf_skip = self.patch_embed.decoder[j](tf_skip)
            expected_shape = skips[-(j+2)].shape
            assert tf_skip.shape == expected_shape, f"TF Skip shape {tf_skip.shape} does not match expected shape {expected_shape}"
            tf_skips.insert(0, tf_skip)     # put in reverse order to match skips
        
        # print("tf_skips:", [x.shape for x in tf_skips])
        # print("skips:", [x.shape for x in skips])
        
        sigWeights = torch.sigmoid(self.skip_weights)
        for i in range(len(skips)):
            if i == 0: continue     # leave skip connection from last encoder block alone
            skips[i] = sigWeights[i-1] * tf_skips[i-1] + (-sigWeights[i-1] + 1) * skips[i]

        # print("x shape before reshape:", x.shape)
        # print("Final x grad_fn:", x.grad_fn)

        # print("Final x shape:", x.shape)
        # print("Final skips shape:", [s.shape for s in skips])
        # print("Final tokens shape:", tokens.shape)
        # print("x == last layer?", torch.all(x == tf_skips[-1]))

        return features, skips, None



if __name__ == "__main__":
    ...
    st_sequential()