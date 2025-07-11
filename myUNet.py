import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from myTransformer import MyTransformer, AttentionPool, ClassifierHead

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

        self.encoder = MyTransformer(expectedChannels, expectedStride, inChannels, num_heads=8)
        self.decoder = pretrainedModelArch.decoder
        self.classifier = torch.nn.Sequential(
            AttentionPool(dim=expectedChannels[-1], heads=4),
            ClassifierHead(dim=expectedChannels[-1]),
        )

        self.ret = "all"

    def forward(self, x: torch.Tensor, metadata: list = None):
        B = x.shape[0]
        imgShape = x.shape[2:]
        features, skips, transformer_tokens = self.encoder(x, metadata)
        skips[-1] = features
        segOut = self.decoder(skips)

        if self.ret == "seg":
            return segOut

        clsOut = self.classifier(transformer_tokens)

        if self.ret == "all":
            return features, segOut, clsOut
        elif self.ret == "cls":
            clsOut = torch.sigmoid(clsOut)     # activate before sending to "segmentation"
            filled = []
            # print("x shape:", x.shape)
            # print("cls_out shape:", cls_out.shape)
            # this is some fuckery. it might work tho...
            for b in range(B):
                fg = torch.fill(torch.empty(imgShape), clsOut[b].item()).to(x.device)
                bg = -fg + 1
                filled.append(torch.stack([bg, fg], dim=0))  # [2, H, W, D]
            filled = torch.stack(filled, dim=0)     # [B, 2, H, W, D]
            # print("cls_out:", clsOut)
            return filled
        return
    
if __name__ == "__main__":
    basepath = r"C:\Users\stoughth\mama-mia-challenge\phase1_submission\Dataset102_BreastTumor\nnUNetTrainer__nnUNetPlans__3d_fullres"
    modelPath = rf"{basepath}\fold_1\checkpoint_final.pth"
    plansPath = rf"{basepath}\plans.json"
    datasetPath = rf"{basepath}\dataset.json"
    model = myUNet(modelPath, plansPath, datasetPath)
    