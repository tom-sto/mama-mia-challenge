import torch
from Transformer import MyTransformerST
from Bottleneck import *
from PCRClassifier import ClassifierHead
from PatchEmbed import PatchEncoder, PatchDecoder
from AttentionPooling import AttentionPooling
import helpers

class MyUNet(torch.nn.Module):
    def __init__(self, 
                 expectedPatchSize: int,
                 expectedChannels: list[int] = [1, 32, 64, 128, 256, 320], 
                 expectedStride: list[int] = [2, 2, 2, 2, 2],
                 pretrainedDecoderPath: str = None,
                 nHeads: int = 8,
                 useSkips: bool = True,
                 joint: bool = True,
                 bottleneck: str = "TransformerST",
                 nBottleneckLayers: int = 4):
        super().__init__()
        
        self.encoder = PatchEncoder(expectedChannels, expectedStride, useSkips=useSkips)
        self.decoder = PatchDecoder(expectedChannels, useSkips)
        self.poolSkips = AttentionPooling(expectedChannels[-1], nHeads)
        
        if pretrainedDecoderPath is not None:
            stateDict: dict = torch.load(pretrainedDecoderPath, map_location='cpu', weights_only=False)['network_weights']

            # Load only decoder weights
            decoderStateDict = {k.replace("decoder.", ""): v for k, v in stateDict.items() if "decoder" in k}
            self.decoder.load_state_dict(decoderStateDict, strict=False)

        self.bottleneckType = bottleneck
        match bottleneck:
            case helpers.BOTTLENECK_TRANSFORMERST:
                self.bottleneck = MyTransformerST(expectedPatchSize, expectedChannels, nHeads, nBottleneckLayers)
            
            # TODO: Implement PCR with these bottlenecks
            case helpers.BOTTLENECK_CONV:
                self.bottleneck = ConvBottleneck(expectedChannels[-1], expectedChannels[-1], nHeads, nBottleneckLayers)
            case helpers.BOTTLENECK_NONE | _:
                self.bottleneck = NoBottleneck(expectedChannels[-1], nHeads)

        self.classifier = ClassifierHead(dim=expectedChannels[-1])
        self.ret = "all" if joint else "seg"

    def forward(self, x: torch.Tensor, patientIDs: list[str], patchIdxs: list[tuple[int]], patientData: list = None):
        x, skips, shape = self.encoder(x)

        x = self.bottleneck(x, shape, patientIDs, patchIdxs)
        if "seg" not in self.ret or "Transformer" in self.bottleneckType:
            sharedFeatures, pcrToken = x    # unpack cls if our bottleneck allows
            x: torch.Tensor = sharedFeatures

        segOut: torch.Tensor = self.decoder(x.reshape(-1, *x.shape[2:]), skips)

        if self.ret == "seg":
            return segOut, None, None
        elif self.ret == "segOnly":
            return segOut

        pcrOut = self.classifier(pcrToken)

        if self.ret == "all":
            return segOut, sharedFeatures, pcrOut
        elif self.ret == "prob":
            return None, None, torch.sigmoid(pcrOut)
        elif self.ret == "probOnly":
            return torch.sigmoid(pcrOut)
        return
    
if __name__ == "__main__":
    basepath = r"C:\Users\stoughth\mama-mia-challenge\phase1_submission\Dataset102_BreastTumor\nnUNetTrainer__nnUNetPlans__3d_fullres"
    modelPath = rf"{basepath}\fold_1\checkpoint_final.pth"
    plansPath = rf"{basepath}\plans.json"
    datasetPath = rf"{basepath}\dataset.json"
    model = MyUNet(modelPath, plansPath, datasetPath)
    