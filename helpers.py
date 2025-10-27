import torch

IMAGE_TYPES = ("phase", "seg", "dmap")
PATCH_SIZE  = 32
NUM_PATCHES = 32
CHUNK_SIZE  = NUM_PATCHES

ACQ_TIME_THRESHOLD = 750
MIN_NUM_PHASES = 3

DTYPE_SEG   = torch.bool
DTYPE_DMAP  = torch.float32
DTYPE_PHASE = torch.float32
DTYPE_PCR   = torch.int32

BOTTLENECK_TRANSFORMERTS = "TransformerTS"
BOTTLENECK_TRANSFORMERST = "TransformerST"
BOTTLENECK_SPATIOTEMPORAL = "SpatioTemporal"
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