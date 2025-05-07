import torch
import numpy as np
import json
import importlib
import os
from dynamic_network_architectures.architectures.unet import PlainConvUNet

from myTransformer import MyTransformer

def resolve_string_to_class(path: str):
    """Convert a string like 'torch.nn.Conv3d' to the actual class."""
    try:
        if not isinstance(path, str):
            return path
        # Basic heuristic: must have at least one dot and no spaces
        if 'torch' not in path:
            return path
        module_path, class_name = path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        # If it fails, return the original string
        print(f"Could not resolve {path}. Returning as string.")
        return path

def resolve_modules_in_dict(d: dict):
    """Recursively convert all string paths to module classes/functions."""
    if isinstance(d, dict):
        return {k: resolve_modules_in_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [resolve_modules_in_dict(i) for i in d]
    elif isinstance(d, tuple):
        return tuple(resolve_modules_in_dict(i) for i in d)
    else:
        return resolve_string_to_class(d)

class myUNet(torch.nn.Module):
    def __init__(self, 
                 pretrainedModelArch: PlainConvUNet,
                 expectedChannels: list[int], 
                 expectedStride: list[int],
                 pretrainedModelPath: str = None,
                 ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if pretrainedModelPath is not None:
            stateDict = torch.load(pretrainedModelPath, map_location='cpu', weights_only=False)['network_weights']

            # Load only decoder weights
            decoderStateDict = {k.replace("decoder.", ""): v for k, v in stateDict.items() if "decoder" in k}
            pretrainedModelArch.decoder.load_state_dict(decoderStateDict, strict=False)

        self.encoder = MyTransformer(expectedChannels, expectedStride)
        self.decoder = pretrainedModelArch.decoder

    def forward(self, x: torch.Tensor):
        print("Incoming shape:", x.shape)
        x = self.encoder(x)
        print("latent shape:", x.shape)
        x = self.decoder(x)
        print("output shape:", x.shape)

        return x
    
if __name__ == "__main__":
    basepath = r"C:\Users\stoughth\mama-mia-challenge\phase1_submission\Dataset102_BreastTumor\nnUNetTrainer__nnUNetPlans__3d_fullres"
    modelPath = rf"{basepath}\fold_1\checkpoint_final.pth"
    plansPath = rf"{basepath}\plans.json"
    datasetPath = rf"{basepath}\dataset.json"
    model = myUNet(modelPath, plansPath, datasetPath)
    