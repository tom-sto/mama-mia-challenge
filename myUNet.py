import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from myTransformer import MyTransformer, ClassifierHead
# from stTransformer import TransformerST
from TransformerTwo import TransformerST

class myUNet(torch.nn.Module):
    def __init__(self, 
                 pretrainedModelArch: PlainConvUNet,
                 inChannels: int,
                 expectedPatchSize: int,
                 expectedChannels: list[int], 
                 expectedStride: list[int],
                 pretrainedModelPath: str = None,
                 p_split: int = 4,
                 n_heads: int = 8
                 ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if pretrainedModelPath is not None:
            stateDict = torch.load(pretrainedModelPath, map_location='cpu', weights_only=False)['network_weights']

            # Load only decoder weights
            decoderStateDict = {k.replace("decoder.", ""): v for k, v in stateDict.items() if "decoder" in k}
            pretrainedModelArch.decoder.load_state_dict(decoderStateDict, strict=False)

        self.encoder = MyTransformer(expectedPatchSize, expectedChannels, expectedStride, inChannels, num_heads=n_heads, p_split=p_split)
        # self.encoder = TransformerST(expectedPatchSize, expectedChannels, expectedStride, inChannels, transformer_num_heads=8, transformer_num_layers=4)
        self.decoder = pretrainedModelArch.decoder
        self.classifier = ClassifierHead(dim=expectedChannels[-1])

        self.ret = "seg"

    def forward(self, x: torch.Tensor, patientData: list = None):
        features, skips, cls_token = self.encoder(x, patientData)
        segOut = self.decoder(skips)

        # print("Features:", features.shape)
        # print("cls token:", cls_token.shape)
        if self.ret == "seg":
            return None, segOut, None

        clsOut = self.classifier(cls_token)

        if self.ret == "all":
            return features, segOut, clsOut
        elif self.ret == "probability":
            return torch.sigmoid(clsOut)
        return
    
if __name__ == "__main__":
    basepath = r"C:\Users\stoughth\mama-mia-challenge\phase1_submission\Dataset102_BreastTumor\nnUNetTrainer__nnUNetPlans__3d_fullres"
    modelPath = rf"{basepath}\fold_1\checkpoint_final.pth"
    plansPath = rf"{basepath}\plans.json"
    datasetPath = rf"{basepath}\dataset.json"
    model = myUNet(modelPath, plansPath, datasetPath)
    