import torch
import torch.nn.functional as F

IMAGE_TYPES = ("phase", "seg", "dmap")
PATCH_SIZE  = 32
NUM_PATCHES = 20
CHUNK_SIZE  = 32

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

EPS = 1e-3

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

# helper function to handle dynamic reshaping
# (..., D, H, W) -> (N_combined, C=1, D, H, W)
def _reshape_to_pyt_format(arr: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    original_shape = arr.shape
    ndim = arr.ndim
    if ndim < 3:
        raise ValueError(f"Input tensor must have at least 3 spatial dimensions (D, H, W). Got {ndim}D.")

    spatial_dims = 3 
    batch_dims = original_shape[:-spatial_dims]
    
    N_combined = torch.prod(torch.tensor(batch_dims)).item() if batch_dims else 1
    reshaped_arr = arr.reshape(N_combined, 1, *original_shape[-spatial_dims:])
    
    return reshaped_arr, batch_dims

def _restore_original_shape(arr_interp: torch.Tensor, original_batch_dims: tuple[int, ...]) -> torch.Tensor:
    arr_no_channel = arr_interp.squeeze(1)
    new_spatial_dims = arr_no_channel.shape[-3:]
    
    if not original_batch_dims:
        # If input was 3D (D, H, W), the output is also 3D (D', H', W')
        return arr_no_channel.squeeze(0)
    else:
        return arr_no_channel.reshape(*original_batch_dims, *new_spatial_dims)

def DownsampleTensor(arr: torch.Tensor, size: int) -> torch.Tensor:
    reshaped_arr, original_batch_dims = _reshape_to_pyt_format(arr)
    newSize = (size, size, size)
    
    downsampled_arr: torch.Tensor = F.interpolate(
        reshaped_arr, 
        size=newSize, 
        mode="trilinear", 
        align_corners=True
    )

    return _restore_original_shape(downsampled_arr, original_batch_dims)

def UpsampleTensor(arr: torch.Tensor, size: int) -> torch.Tensor:
    reshaped_arr, original_batch_dims = _reshape_to_pyt_format(arr)
    newSize = (size, size, size)

    upsampled_arr: torch.Tensor = F.interpolate(
        reshaped_arr, 
        size=newSize, 
        mode="trilinear", 
        align_corners=True
    )

    return _restore_original_shape(upsampled_arr, original_batch_dims)