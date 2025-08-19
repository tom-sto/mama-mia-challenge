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
        # print("Called attention pooling!")
        # x: [B, N, T, E]
        B, N, T, E = x.shape
        H, d = self.num_heads, self.head_dim

        # Project inputs to keys and values
        q: torch.Tensor = self.query_proj(self.q_cls.repeat(B, N, 1, 1))   # [B, N, 1, E] 
        k: torch.Tensor = self.key_proj(x)              # [B, N, T, E]
        v: torch.Tensor = self.value_proj(x)            # [B, N, T, E]
        # print(f"\tq: {q.shape}\tk: {k.shape}\tv: {v.shape}")

        # Reshape for multi-head: [B, H, T, d]
        q = q.view(B, N, 1, H, d).transpose(2, 3)       # [B, N, H, 1, d]
        k = k.view(B, N, T, H, d).transpose(2, 3)       # [B, N, H, T, d]
        v = v.view(B, N, T, H, d).transpose(2, 3)       # [B, N, H, T, d]
        # print(f"\tq: {q.shape}\tk: {k.shape}\tv: {v.shape}")

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)        # [B, N, H, 1, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)           # [B, N, H, 1, T]
        pooled = attn_weights @ v                                   # [B, N, H, 1, d]
        # print(f"\tpooled weights before squeeze {pooled.shape}")

        pooled = pooled.squeeze(3).reshape(B, N, E)                 # [B, N, E]

        # Final projection
        return self.out_proj(pooled)