import torch, torch.nn as nn

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
        # x: [..., N, E]
        N, E = x.shape[-2:]
        D = x.shape[:-2]
        H, d = self.num_heads, self.head_dim

        # Project inputs to keys and values
        q: torch.Tensor = self.query_proj(self.q_cls.repeat(*D, 1, 1))   # [..., 1, E] 
        k: torch.Tensor = self.key_proj(x)              # [..., N, E]
        v: torch.Tensor = self.value_proj(x)            # [..., N, E]

        # Reshape for multi-head: [B, H, T, d]
        q = q.view(*D, 1, H, d).transpose(-2, -3)       # [..., H, 1, d]
        k = k.view(*D, N, H, d).transpose(-2, -3)       # [..., H, N, d]
        v = v.view(*D, N, H, d).transpose(-2, -3)       # [..., H, N, d]

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)        # [B, N, H, 1, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)           # [B, N, H, 1, T]
        pooled = attn_weights @ v                                   # [B, N, H, 1, d]

        pooled = pooled.squeeze(3).reshape(*D, E)                 # [..., E]

        # Final projection
        return self.out_proj(pooled)