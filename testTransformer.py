import torch
from TransformerTwo import AttentionPooling

def ts_sequential():
    n_heads = 2
    n_patches = 32
    n_phases = 6
    emb_dim = 20
    batch_size = 3

    poolMe = AttentionPooling(emb_dim, num_heads=n_heads)

    d = emb_dim // n_heads
    assert emb_dim % n_heads == 0, "Need embedding dim to be divisible by the number of heads"

    # simulate embedding matrix after encoding, patient data embedding
    emb = torch.randn(batch_size, n_patches, n_phases, emb_dim)

    W_qkv_s = torch.nn.Linear(emb_dim, emb_dim * 3)
    W_os = torch.nn.Linear(emb_dim, emb_dim)

    W_qkv_t = torch.nn.Linear(emb_dim, emb_dim * 3)
    W_ot = torch.nn.Linear(emb_dim, emb_dim)

    print(f"emb: {emb.shape}")
    x = emb

    B, N, T = batch_size, n_patches, n_phases

    qkv: torch.Tensor = W_qkv_t(x)
    qkv = qkv.reshape(B, N, T, 3, n_heads, d)          # [B, N, T, 3, H, d]
    qkv = qkv.permute(3, 0, 1, 4, 2, 5)                # [3, B, N, H, T, d]
    Q, K, V = qkv[0], qkv[1], qkv[2]                # Each: [B, H, T, d]

    print(f"\tQ shape: {Q.shape}")
    print(f"\tK shape: {K.shape}")
    print(f"\tV shape: {V.shape}")

    # Compute attention
    scores = (Q @ K.transpose(-2, -1)) / (d ** 0.5)
    print(f"\tScores shape: {scores.shape}")        # (B, N, h, T, T)

    attn = torch.softmax(scores, dim=-1)
    print(f"\tattn shape: {attn.shape}")            # (B, N, h, T, T)

    context = attn @ V                              # (B, N, h, T, d)
    print(f"\tcontext shape: {context.shape}")

    # Concatenate heads
    context = context.permute(0, 1, 3, 2, 4).contiguous()
    context = context.view(B, N, T, -1)
    print(f"\tcontext shape (after concat): {context.shape}")

    O: torch.Tensor = W_ot(context)
    print(f"\tO shape = {O.shape}")

    x += O
    x = torch.layer_norm(x, normalized_shape=(emb_dim,))
    print(f"\tFinal x shape: {x.shape}")

    # attention pooling
    x: torch.Tensor = poolMe(x)
    print(f"Spatial transformer:")

    qkv: torch.Tensor = W_qkv_s(x)
    qkv = qkv.reshape(B, N, 3, n_heads, d)          # [B, N, 3, H, d]
    qkv = qkv.permute(2, 0, 1, 3, 4)                # [3, B, H, N, d]
    Q, K, V = qkv[0], qkv[1], qkv[2]                # Each: [B, H, N, d]

    print(f"\tQ shape: {Q.shape}")
    print(f"\tK shape: {K.shape}")
    print(f"\tV shape: {V.shape}")

    # Compute attention
    scores = (Q @ K.transpose(-2, -1)) / (d ** 0.5)
    print(f"\tScores shape: {scores.shape}")        # (B, h, N, N)

    attn = torch.softmax(scores, dim=-1)
    print(f"\tattn shape: {attn.shape}")            # (B, h, N, N)

    context = attn @ V                              # (B, h, N, d)
    print(f"\tcontext shape: {context.shape}")

    # Concatenate heads
    context = context.permute(0, 2, 1, 3).contiguous()
    context = context.view(B, N, -1)
    print(f"\tcontext shape (after concat): {context.shape}")

    O: torch.Tensor = W_os(context)
    print(f"\tO shape = {O.shape}")

    x += O
    x = torch.layer_norm(x, normalized_shape=(emb_dim,))
    print(f"\tFinal x shape: {x.shape}")


if __name__ == "__main__":
    ts_sequential()