import torch, torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Learned query vector
        self.q_cls = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.q_cls, std=0.02)

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        # x: [..., N, E]
        N, E = x.shape[-2:]
        D = x.shape[:-2]
        H, d = self.num_heads, self.head_dim

        # Project inputs to keys and values
        q: torch.Tensor = self.query_proj(self.q_cls.expand(*D, 1, E))   # [..., 1, E] 
        k: torch.Tensor = self.key_proj(x)                          # [..., N, E]
        v: torch.Tensor = self.value_proj(x)                        # [..., N, E]
            
        # Reshape for multi-head: [B, H, T, d]          
        q = q.view(*D, 1, H, d).transpose(-2, -3).contiguous()      # [..., H, 1, d]
        k = k.view(*D, N, H, d).transpose(-2, -3).contiguous()      # [..., H, N, d]
        v = v.view(*D, N, H, d).transpose(-2, -3).contiguous()      # [..., H, N, d]

        # Scaled dot-product attention
        pooled = nn.functional.scaled_dot_product_attention(q, k, v)

        # Reshape back to [..., E]
        pooled = pooled.transpose(-2, -3).reshape(*D, E)            # [B, N, E]

        # Final projection
        return self.out_proj(pooled)