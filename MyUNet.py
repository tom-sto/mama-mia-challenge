import torch
from Transformer import MyTransformerTS, MyTransformerST, MySpatioTemporalTransformer
from Bottleneck import *
from PCRClassifier import ClassifierHead, ClassifierHeadWithConfidence
from PatchEmbed import PatchEncoder, PatchDecoder
from AttentionPooling import AttentionPooling
import helpers
from Encodings import PositionEncoding3D

class MyUNet(torch.nn.Module):
    def __init__(self, 
                 expectedPatchSize: int,
                 expectedChannels: list[int], 
                 expectedStride: list[int] = [2, 2, 2, 2, 2],
                 pretrainedDecoderPath: str = None,
                 patientDataPath: str = None,
                 nHeads: int = 8,
                 useSkips: bool = True,
                 joint: bool = True,
                 catPosDecoder: bool = True,
                 pcrConfidence: bool = False,
                 bottleneck: str = "TransformerST",
                 nBottleneckLayers: int = 4,
                 useAttentionPooling: bool = True):
        super().__init__()

        # [1, 64, 128, 256, 384, 576]
        # if not useSkips:
        #     # [1, 96, 192, 384, 576, 864]
        #     expectedChannels = [1] + [round(i * 1.5) for i in expectedChannels[1:]]
            
        
        self.encoder = PatchEncoder(expectedChannels, expectedStride, dropout=0, useSkips=useSkips)
        self.decoder = PatchDecoder(expectedChannels, catPosDecoder, useSkips=useSkips)
        self.poolSkips = AttentionPooling(expectedChannels[-1], nHeads)
        
        if pretrainedDecoderPath is not None:
            stateDict: dict = torch.load(pretrainedDecoderPath, map_location='cpu', weights_only=False)['networkWeights']

            # Load only decoder weights
            decoderStateDict = {k.replace("decoder.", ""): v for k, v in stateDict.items() if "decoder" in k}
            self.decoder.load_state_dict(decoderStateDict, strict=False)

        self.bottleneckType = bottleneck
        match bottleneck:
            case helpers.BOTTLENECK_TRANSFORMERTS:
                self.bottleneck = MyTransformerTS(expectedPatchSize, expectedChannels, nHeads, nBottleneckLayers, patientDataPath)
            case helpers.BOTTLENECK_TRANSFORMERST:
                self.bottleneck = MyTransformerST(expectedPatchSize, expectedChannels, nHeads, nBottleneckLayers, patientDataPath)
            case helpers.BOTTLENECK_SPATIOTEMPORAL:
                self.bottleneck = MySpatioTemporalTransformer(expectedPatchSize, expectedChannels, nHeads, nBottleneckLayers, patientDataPath, useAttentionPooling=useAttentionPooling)
            
            # TODO: Implement PCR with these bottlenecks
            case helpers.BOTTLENECK_CONV:
                self.bottleneck = ConvBottleneck(expectedChannels[-1], expectedChannels[-1], nHeads, nBottleneckLayers)
            case helpers.BOTTLENECK_NONE | _:
                self.bottleneck = NoBottleneck(expectedChannels[-1], nHeads)

        self.classifier = ClassifierHead(dim=expectedChannels[-1]) if not pcrConfidence else ClassifierHeadWithConfidence(dim=expectedChannels[-1])
        self.catPosDecoder = catPosDecoder
        self.ret = "all" if joint else "seg"

    def forward(self, x: torch.Tensor, patientIDs: list[str], patchIdxs: torch.Tensor, patientData: list = None):
        x, skips, shape = self.encoder(x)
        B, N = shape[0], shape[2]

        sharedFeatures: torch.Tensor = self.bottleneck(x, shape, patientIDs, patchIdxs)    # [B, E]
        E = sharedFeatures.shape[-1]
        pcrOut: torch.Tensor = self.classifier(sharedFeatures)       # [B, 1]

        posEnc = PositionEncoding3D(patchIdxs, dim=E)     # [B, N, E]
        if self.catPosDecoder:
            x = torch.cat([posEnc, sharedFeatures.unsqueeze(1).repeat(1, N, 1)], dim=-1)
            x = x.reshape(-1, 2*E)[..., None, None, None]            # [B*N, E, 1, 1, 1]
        else:
            x = posEnc + sharedFeatures.unsqueeze(1).repeat(1, N, 1)
            x = x.reshape(-1, E)[..., None, None, None]
        segOut: torch.Tensor = self.decoder(x, skips)
        segOut = segOut.reshape(B, N, *segOut.shape[-3:])

        if self.ret == "seg":
            return segOut, None, None
        elif self.ret == "segOnly":
            return segOut

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
    