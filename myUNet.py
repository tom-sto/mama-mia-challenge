import torch
from Transformer import MyTransformerST
from Bottleneck import *
from PCRClassifier import ClassifierHead
from PatchEmbed import PatchEncoder, PatchDecoder
from AttentionPooling import AttentionPooling

class MyUNet(torch.nn.Module):
    def __init__(self, 
                 expectedPatchSize: int,
                 expectedChannels: list[int] = [1, 4, 16, 48, 128, 256], 
                 expectedStride: list[int] = [2, 2, 2, 2, 2, 2],
                 pretrainedDecoderPath: str = None,
                 nHeads: int = 8,
                 useSkips: bool = True,
                 bottleneck: str = "TransformerST",
                 nBottleneckLayers: int = 4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encoder = PatchEncoder(expectedChannels, expectedStride, useSkips=useSkips)
        self.decoder = PatchDecoder(expectedChannels, useSkips)
        self.poolSkips = AttentionPooling(expectedChannels[-1], nHeads)
        
        if pretrainedDecoderPath is not None:
            stateDict: dict = torch.load(pretrainedDecoderPath, map_location='cpu', weights_only=False)['network_weights']

            # Load only decoder weights
            decoderStateDict = {k.replace("decoder.", ""): v for k, v in stateDict.items() if "decoder" in k}
            self.decoder.load_state_dict(decoderStateDict, strict=False)

        match bottleneck:
            case "TransformerST":
                self.bottleneck = MyTransformerST(expectedPatchSize, 
                                                  expectedChannels, 
                                                  nHeads=nHeads, 
                                                  transformer_num_layers=4)
            case "Conv":
                self.bottleneck = ConvBottleneck(expectedChannels[-1], 
                                                 expectedChannels[-1], 
                                                 nBottleneckLayers, 
                                                 nHeads)
            case "None" | _:
                self.bottleneck = NoBottleneck(expectedChannels[-1], nHeads)
        self.classifier = ClassifierHead(dim=expectedChannels[-1])

        self.ret = "all"

    def forward(self, x: torch.Tensor, patientIDs: list[str], patchIdxs: list[tuple[int]], patientData: list = None):
        x, skips, shape = self.encoder(x)
        # print(f"shape after enc: {x.shape}")
        # print(f"skips: {[skips[i].shape for i in range(len(skips))]}")

        x = self.bottleneck(x, shape, patientIDs, patchIdxs)
        if "seg" not in self.ret:
            sharedFeatures, clsToken = x    # unpack cls if our bottleneck allows
            x: torch.Tensor = sharedFeatures
        else:
            x, _ = x
        # print(f"shape after bottleneck: {x.shape}")
        # print(f"cls token shape: {clsToken.shape}")

        # print(f"Bottleneck output has nan: {torch.any(x.isnan())}")
        # print(f"\t{x.min(), x.max()}")
        # put batch and patches together for decoder
        segOut: torch.Tensor = self.decoder(x.reshape(-1, *x.shape[2:]), skips)
        # print(f"seg out shape: {segOut.shape}")
        # print(f"Decoder output has nan: {torch.any(segOut.isnan())}")
        # print(f"\t{segOut.min(), segOut.max()}")
        # print(f"target shape: {target.shape}")

        # print("Features:", features.shape)
        # print("cls token:", cls_token.shape)
        if self.ret == "seg":
            return segOut, None, None
        elif self.ret == "segOnly":
            return segOut

        pcrOut = self.classifier(clsToken)

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
    