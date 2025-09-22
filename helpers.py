import torch
import pandas as pd
import numba

IMAGE_TYPES = ("phase", "seg", "dmap")
PATCH_SIZE  = 32
CHUNK_SIZE  = 64
NUM_PATCHES = 64

ACQ_TIME_THRESHOLD = 750
MIN_NUM_PHASES = 3

DTYPE_SEG   = torch.bool
DTYPE_DMAP  = torch.float16
DTYPE_PHASE = torch.float16

BOTTLENECK_TRANSFORMERST = "TransformerST"
BOTTLENECK_CONV = "Conv"
BOTTLENECK_NONE = "None"

EPS = 1e-5

def Mean(l: list):
    if len(l) == 0:
        return -1
    return sum(l) / len(l)

def FormatSeconds(s):
    if s < 60:
        return f"{s:.4f} seconds"
    minutes = s // 60
    seconds = s % 60
    if s < 3600:
        return f"{minutes:.0f} minute{"s" if minutes > 1 else ""} and {seconds:.4f} seconds"
    hours = minutes // 60
    minutes %= 60
    if s < 86400:
        return f"{hours:.0f} hour{"s" if hours > 1 else ""}, {minutes:.0f} minute{"s" if minutes > 1 else ""}, and {seconds:.4f} seconds"
    # should never take this long, but you never know...
    days = hours // 24
    hours %= 24
    return f"{days:.0f} day{"s" if days > 1 else ""}, {hours:.0f} hour{"s" if hours > 1 else ""}, {minutes:.0f} minute{"s" if minutes > 1 else ""}, and {seconds:.4f} seconds"

# adapted from PE formula in Viswani et. al. (2023)
def PositionEncoding(seq: torch.Tensor, dim: int, div=10_000, scale=None):
    if scale is not None:
        max_vals = seq.max(dim=-1, keepdim=True).values
        seq = seq*scale / max_vals.clamp(min=1e-8)
    position = seq.unsqueeze(-1)                                # [..., N, 1]

    i = torch.arange(0, dim, 2, device=position.device)                  # [E/2]
    div_term = 1.0 / (div ** (i / dim))

    pos = position * div_term                                   # [..., N, E/2]
    pos_enc = torch.empty(*seq.shape, dim, device=position.device)       # [..., N, E]
    pos_enc[..., 0::2] = torch.sin(pos)                         # [..., N, E/2] for even emb dims
    pos_enc[..., 1::2] = torch.cos(pos)                         # [..., N, E/2] for odd emb dims
    
    return pos_enc                                              # [..., N, E]

def PositionEncoding3D(seq: torch.Tensor, dim: int):
    # since we want the embedding dim to not be super constrained, we split the dims as evenly as we can
    # give the z-axis the short end of the stick if need be, then y-axis

    x, y, z = seq[..., 0], seq[..., 1], seq[..., 2]
    d = (dim // 2) // 3
    dX, dY, dZ = 2*d, 2*d, 2*d
    if dim % (d * 6) == 4:
        dX += 2
        dY += 2
    elif dim % (d * 6) == 2:
        dX += 2

    assert dX + dY + dZ == dim, f"You did the math wrong dummy! {dX} + {dY} + {dZ} != {dim}"
    assert not any([dX % 2, dY % 2, dZ % 2]), f"Need these dims to be even!, Got ({dX}, {dY}, {dZ})"
    
    xEnc = PositionEncoding(x, dim=dX)
    yEnc = PositionEncoding(y, dim=dY)
    zEnc = PositionEncoding(z, dim=dZ)

    pos_enc = torch.cat([xEnc, yEnc, zEnc], dim=-1)
    return pos_enc

def CleanPatientData(df: pd.DataFrame, 
                     patient_ids: list[str], 
                     columns: list[str] = ["age", 
                                           "menopause",
                                           "breast_density",
                                           "acquisition_times"]):
    def cleanMenopause(df: pd.DataFrame):
        df['menopause'] = df['menopause'].fillna('unknown')
        df['menopause'] = df['menopause'].apply(lambda x: 'pre' if 'peri' in x else x)
        df['menopause'] = df['menopause'].apply(lambda x: 'post' if 'post' in x else x)
        df['menopause'] = df['menopause'].apply(lambda x: 'pre' if 'pre' in x else x)

        return df

    df = cleanMenopause(df)

    data = []
    for pid in patient_ids:
        md = {}
        for m in columns:
            d = df.loc[df["patient_id"] == pid.upper(), m].item()
            if pd.isna(d):
                d = None
            elif m == "acquisition_times":
                d = eval(d)
            md[m] = d
        data.append(md)

    # data = [{m: df.loc[df["patient_id"] == pid.upper(), m] for m in columns} for pid in patient_ids]

    return data