import torch
import torch.nn as nn
from math import log
import copy

class TransformerLayerT(nn.Module):
    def __init__(self,
                 emb_dim,
                 n_heads,
                 dropout = 0.1,
                 layer_norm_eps: float = 1e-5):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.d = self.emb_dim // self.n_heads

        dim_feedforward = emb_dim * 4

        self.W_qkv      = nn.Linear(emb_dim, emb_dim * 3)
        self.W_o        = nn.Linear(emb_dim, emb_dim)
        self.linear1    = nn.Linear(emb_dim, dim_feedforward)
        self.dropout    = nn.Dropout(dropout)
        self.linear2    = nn.Linear(dim_feedforward, emb_dim)
        self.activation = nn.ReLU()

        self.norm1      = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.norm2      = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        print(f"Temporal transformer:")
        B, N, T, _ = x.shape
        
        qkv: torch.Tensor = self.W_qkv(x)
        qkv = qkv.reshape(B, N, T, 3, self.n_heads, self.d)          # [B, N, T, 3, H, d]
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)                # [3, B, N, H, T, d]
        Q, K, V = qkv[0], qkv[1], qkv[2]                # Each: [B, H, T, d]

        print(f"\tQ shape: {Q.shape}")
        print(f"\tK shape: {K.shape}")
        print(f"\tV shape: {V.shape}")

        # Compute attention
        scores = (Q @ K.transpose(-2, -1)) / (self.d ** 0.5)
        print(f"\tScores shape: {scores.shape}")        # (B, N, h, T, T)

        attn = torch.softmax(scores, dim=-1)
        print(f"\tattn shape: {attn.shape}")            # (B, N, h, T, T)

        context = attn @ V                              # (B, N, h, T, d)
        print(f"\tcontext shape: {context.shape}")

        # Concatenate heads
        context = context.permute(0, 1, 3, 2, 4).contiguous()
        context = context.view(B, N, T, -1)
        print(f"\tcontext shape (after concat): {context.shape}")

        O: torch.Tensor = self.W_o(context)
        print(f"\tO shape = {O.shape}")

        x += O
        x = self.norm1(x)
        x = self.dropout1(x)

        # FFN
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x))))    # Linear -> ReLU -> Dropout -> Linear
        ffn_out = self.dropout2(ffn_out)

        x += ffn_out
        x = self.norm2(x)

        return x

class TransformerLayerS(nn.Module):
    def __init__(self,
                 emb_dim,
                 n_heads,
                 dropout = 0.1,
                 layer_norm_eps: float = 1e-5):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.d = self.emb_dim // self.n_heads

        dim_feedforward = emb_dim * 4

        self.W_qkv      = nn.Linear(emb_dim, emb_dim * 3)
        self.W_o        = nn.Linear(emb_dim, emb_dim)
        self.linear1    = nn.Linear(emb_dim, dim_feedforward)
        self.dropout    = nn.Dropout(dropout)
        self.linear2    = nn.Linear(dim_feedforward, emb_dim)
        self.activation = nn.ReLU()

        self.norm1      = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.norm2      = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor):
        print(f"Spatial transformer:")
        B, N, _ = x.shape
        
        qkv: torch.Tensor = self.W_qkv(x)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d)          # [B, N, 3, H, d]
        qkv = qkv.permute(2, 0, 1, 3, 4)                # [3, B, H, N, d]
        Q, K, V = qkv[0], qkv[1], qkv[2]                # Each: [B, H, N, d]

        print(f"\tQ shape: {Q.shape}")
        print(f"\tK shape: {K.shape}")
        print(f"\tV shape: {V.shape}")

        # Compute attention
        scores = (Q @ K.transpose(-2, -1)) / (self.d ** 0.5)
        print(f"\tScores shape: {scores.shape}")        # (B, h, N, N)

        attn = torch.softmax(scores, dim=-1)
        print(f"\tattn shape: {attn.shape}")            # (B, h, N, N)

        context = attn @ V                              # (B, h, N, d)
        print(f"\tcontext shape: {context.shape}")

        # Concatenate heads
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(B, N, -1)
        print(f"\tcontext shape (after concat): {context.shape}")

        O: torch.Tensor = self.W_o(context)
        print(f"\tO shape = {O.shape}")

        x += O
        x = self.norm1(x)
        x = self.dropout1(x)

        # FFN
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x))))    # Linear -> ReLU -> Dropout -> Linear
        ffn_out = self.dropout2(ffn_out)

        x += ffn_out
        x = self.norm2(x)

        return x
    
class Transformer(nn.Module):
    def __init__(self,
                 encoder_layer,
                 num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerST(nn.Module):
    def __init__(self,
                 patch_size,
                 channels,
                 strides, 
                 num_heads=4,
                 transformer_num_layers=6,
                 p_split=4):
        super().__init__()

        self.patch_size = patch_size
        self.p_split = p_split

        num_patches = round(self.p_split**3)
        self.num_patches = num_patches

        self.emb_dim = channels[-1]

        nPatientDataInFeatures = 7             # 1 for age (linear), 2 for menopausal status (one-hot), 4 for breast density (one-hot)
        self.nPatientDataOutFeatures = num_heads         # needs to be divisible by num_heads
        self.patientDataEmbed = nn.Linear(nPatientDataInFeatures, self.nPatientDataOutFeatures)

        expectedXYZ = patch_size
        for _ in range(len(channels) + 1):
            expectedXYZ = max(round(expectedXYZ / 2), 1)

        self.patch_embed = DeepPatchEmbed3D(channels, strides)
        self.cls_token   = nn.Parameter(torch.randn((1, 1, 1, self.emb_dim + self.nPatientDataOutFeatures)))
        
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

        tLayer = TransformerLayerT(emb_dim=self.emb_dim + self.nPatientDataOutFeatures, n_heads=num_heads)
        sLayer = TransformerLayerS(emb_dim=self.emb_dim + self.nPatientDataOutFeatures, n_heads=num_heads)

        self.transformerT   = Transformer(tLayer, num_layers=transformer_num_layers)
        self.temporal_proj  = AttentionPooling(self.emb_dim + self.nPatientDataOutFeatures, num_heads)
        self.transformerS   = Transformer(sLayer, num_layers=transformer_num_layers)
        self.fc_to_patches = nn.Linear(self.emb_dim + self.nPatientDataOutFeatures, self.emb_dim)

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

    def get_sinusoid_encoding_table(self, length, dim, device):
        position = torch.arange(length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * -(log(10000.0) / dim))
        sinusoid = torch.zeros(length, dim, device=device)
        sinusoid[:, 0::2] = torch.sin(position * div_term)
        sinusoid[:, 1::2] = torch.cos(position * div_term)
        return sinusoid  # [length, dim]
    
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
        x = self.temporal_proj(x)                                                   # [B*(N*X*Y*Z + 1), E + npatientDataOutFeatures]
        x = x.reshape(B, N*X*Y*Z + 1, E + self.nPatientDataOutFeatures)             # [B, N*X*Y*Z + 1, E + npatientDataOutFeatures]

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

class DeepPatchEmbed3D(nn.Module):
    def __init__(self, channels: list[int], strides: list[int], dropout: float = 0.2):
        super().__init__()
        channels.insert(0, 1)
        self.channels = channels
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        numBlocks = len(channels) - 1

        # use the same encoder block design as my autoencoder! that worked pretty well!
        # we can use batch norm even if the batch size is 1 because we artifically inflate it with the patches
        for i in range(numBlocks):
            block = nn.Sequential(
                nn.Dropout3d(dropout) if i == numBlocks - 1 else nn.Identity(),
                nn.BatchNorm3d(channels[i]),
                nn.Conv3d(channels[i], channels[i], kernel_size=3, padding=1) if i == 0 or i == 1 else nn.Identity(),
                nn.Conv3d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                # nn.BatchNorm3d(channels[i+1]),
                nn.ReLU(True),
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

        skips: list[torch.Tensor] = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            unpatched = unpatchTensor(x, nSubPatches)

            _, E, X, Y, Z = unpatched.shape
            unpatched = unpatched.reshape(B, T, E, X, Y, Z)
            unpatched = unpatched.mean(dim=1)       # for skip connections, reduce phases with mean
            skips.append(unpatched)

        _, E, X, Y, Z = x.shape
        assert X == Y == Z == 1, f"Expected spatial dims to reduce to 1, got {X} x {Y} x {Z}"
        x = x.reshape(B, T, N, E, X, Y, Z)
        x = x.permute(0, 2, 4, 5, 6, 1, 3)      # [B, N, X, Y, Z, T, E] -> put N, X, Y, Z next to each other so they can be squished
        x = x.reshape(B, -1, T, E)              # [B, N*X*Y*Z, T, E]

        return x, skips, (B, T, N, E, X, Y, Z)

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Learned query vector
        self.q_cls = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # Shared linear projections across heads
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Optional output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        print("Called attention pooling!")
        # x: [B, N, T, E]
        B, N, T, E = x.shape
        H, d = self.num_heads, self.head_dim

        # Project inputs to keys and values
        q: torch.Tensor = self.query_proj(self.q_cls.repeat(B, N, 1, 1))   # [B, N, 1, E] 
        k: torch.Tensor = self.key_proj(x)              # [B, N, T, E]
        v: torch.Tensor = self.value_proj(x)            # [B, N, T, E]
        print(f"\tq: {q.shape}\tk: {k.shape}\tv: {v.shape}")

        # Reshape for multi-head: [B, H, T, d]
        q = q.view(B, N, 1, H, d).transpose(2, 3)       # [B, N, H, 1, d]
        k = k.view(B, N, T, H, d).transpose(2, 3)       # [B, N, H, T, d]
        v = v.view(B, N, T, H, d).transpose(2, 3)       # [B, N, H, T, d]
        print(f"\tq: {q.shape}\tk: {k.shape}\tv: {v.shape}")

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)        # [B, N, H, 1, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)           # [B, N, H, 1, T]
        pooled = attn_weights @ v                                   # [B, N, H, 1, d]
        print(f"\tpooled weights before squeeze {pooled.shape}")

        # Collapse attention output: [B, N, H, 1, d] → [B, E]
        pooled = pooled.squeeze(3).reshape(B, N, E)

        # Final projection
        return self.out_proj(pooled)  # [B, E]

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

if __name__ == "__main__":
    ...