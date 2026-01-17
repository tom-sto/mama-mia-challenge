import torch
from Transformer import MyTransformerTS, MyTransformerST, MySpatioTemporalTransformer
from Bottleneck import *
from PCRClassifier import ClassifierHead
from PatchEmbed import PatchEncoder, PatchDecoder
import helpers
from Encodings import PatientDataEncoding, PositionEncoding3D

class MyUNet(torch.nn.Module):
    def __init__(self, 
                 expectedPatchSize: int,
                 expectedChannels: list[int], 
                 expectedStride: list[int] = [2, 2, 2, 2, 2],
                 pretrainedDecoderPath: str = None,
                 patientDataPath: str = None,
                 nHeads: int = 8,
                 useSkips: bool = True,
                 mode: int = 0,
                 catPosDecoder: int = None,
                 bottleneck: str = "TransformerST",
                 nBottleneckLayers: int = 4,
                 useAttentionPooling: bool = True):
        super().__init__()
        self.mode = mode

        # [1, 64, 128, 256, 384, 576]
        if not useSkips:
            # [1, 96, 192, 384, 576, 864]
            expectedChannels = [1] + [round(i * 1.5) for i in expectedChannels[1:]]
        
        self.encoder = PatchEncoder(expectedChannels, expectedStride, dropout=0, useSkips=useSkips)
        self.decoder = PatchDecoder(expectedChannels, catPosDecoder, useSkips=useSkips)
        self.patientDataMod = PatientDataEncoding(patientDataPath, expectedChannels[-1]*2)
        
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
                self.bottleneck = MySpatioTemporalTransformer(expectedPatchSize, expectedChannels, nHeads, nBottleneckLayers, self.patientDataMod, useAttentionPooling=useAttentionPooling)
            
            # TODO: Implement PCR with these bottlenecks
            case helpers.BOTTLENECK_CONV:
                self.bottleneck = ConvBottleneck(expectedChannels[-1], expectedChannels[-1], nHeads, nBottleneckLayers)
            case helpers.BOTTLENECK_NONE | _:
                self.bottleneck = NoBottleneck(expectedChannels[-1], nHeads)
        # self.targetLatentAgg = TargetLatentAgg(dim=expectedChannels[-1], out=helpers.NUM_PATCHES)
        self.classifier = ClassifierHead(dim=expectedChannels[-1]*2 + self.patientDataMod.outFeatures)
        self.catPosDecoder = catPosDecoder

    def forward(self, mri: torch.Tensor, patientIDs: list[str], patchIdxs: torch.Tensor, target: torch.Tensor):
        x, skips, shape = self.encoder(mri)
        B, N = shape[0], shape[2]

        acqTimes = self.patientDataMod.AcquisitionTimes(shape, patientIDs, mri.device)
        latent: torch.Tensor = self.bottleneck(x, shape, patchIdxs, acqTimes)    # [B, E]
        E = latent.shape[-1]

        if self.mode == helpers.MODE_PCR:
            y, _, yShape = self.encoder(target.unsqueeze(1))                 # [B, 1, N, E, X, Y, Z]

            acqTimes = self.patientDataMod.AcquisitionTimes(yShape, patientIDs, mri.device)
            targetLatent: torch.Tensor = self.bottleneck(y, yShape, patchIdxs, acqTimes)            # [B, E]
            latent = torch.cat([latent, targetLatent], dim=-1)                                      # [B, 2*E]
            patientDataEmb, rfpLoss = self.patientDataMod(latent, patientIDs)                       # [B, R]
            pcrOut: torch.Tensor = self.classifier(torch.cat([latent, patientDataEmb], dim=-1))     # [B, 2*E + R] -> [B, 1]
            return pcrOut, rfpLoss

        if self.catPosDecoder is not None:
            posEnc = PositionEncoding3D(patchIdxs, dim=self.catPosDecoder)              # [B, N, C]
            x = torch.cat([posEnc, latent.unsqueeze(1).repeat(1, N, 1)], dim=-1)
            x = x.reshape(-1, E + self.catPosDecoder)[..., None, None, None]            # [B*N, E + C, 1, 1, 1]
        else:
            posEnc = PositionEncoding3D(patchIdxs, dim=E)           # [B, N, E]
            x = posEnc + latent.unsqueeze(1).repeat(1, N, 1)
            x = x.reshape(-1, E)[..., None, None, None]             # [B*N, E, 1, 1, 1]
        segOut: torch.Tensor = self.decoder(x, skips)
        segOut = segOut.reshape(B, N, *segOut.shape[-3:])

        return segOut
    
if __name__ == "__main__":
    basepath = r"C:\Users\stoughth\mama-mia-challenge\phase1_submission\Dataset102_BreastTumor\nnUNetTrainer__nnUNetPlans__3d_fullres"
    modelPath = rf"{basepath}\fold_1\checkpoint_final.pth"
    plansPath = rf"{basepath}\plans.json"
    datasetPath = rf"{basepath}\dataset.json"
    model = MyUNet(modelPath, plansPath, datasetPath)
    