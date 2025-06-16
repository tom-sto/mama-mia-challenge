import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from myTransformer import MyTransformer

class myUNet(torch.nn.Module):
    def __init__(self, 
                 pretrainedModelArch: PlainConvUNet,
                 inChannels: int,
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

        self.encoder = MyTransformer(expectedChannels, expectedStride, inChannels)
        self.decoder = pretrainedModelArch.decoder

    def forward(self, x: torch.Tensor, metadata: list = None):
        x, skips = self.encoder(x, metadata)
        skips[-1] = x
        x = self.decoder(skips)

        return x
    
if __name__ == "__main__":
    basepath = r"C:\Users\stoughth\mama-mia-challenge\phase1_submission\Dataset102_BreastTumor\nnUNetTrainer__nnUNetPlans__3d_fullres"
    modelPath = rf"{basepath}\fold_1\checkpoint_final.pth"
    plansPath = rf"{basepath}\plans.json"
    datasetPath = rf"{basepath}\dataset.json"
    model = myUNet(modelPath, plansPath, datasetPath)
    