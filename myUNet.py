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

        self.encoder = MyTransformer(expectedChannels, expectedStride, inChannels)
        self.decoder = pretrainedModelArch.decoder

        self.attn_pool = AttentionPool(dim=expectedChannels[-1], heads=4)
        self.classifier_head = ClassifierHead(dim=expectedChannels[-1])

        self.ret = "both"

    def forward(self, x: torch.Tensor, metadata: list = None):
        B = x.shape[0]
        imgShape = x.shape[2:]
        x, skips, transformer_tokens = self.encoder(x, metadata)
        skips[-1] = x
        x = self.decoder(skips)

        if self.ret == "seg":
            return x

        pooled = self.attn_pool(transformer_tokens)
        cls_out = self.classifier_head(pooled)

        if self.ret == "both":
            return x, cls_out
        elif self.ret == "cls":
            cls_out = torch.sigmoid(cls_out)     # activate before sending to "segmentation"
            filled = []
            # print("x shape:", x.shape)
            # print("cls_out shape:", cls_out.shape)
            # this is some fuckery. it might work tho...
            for b in range(B):
                fg = torch.fill(torch.empty(imgShape), cls_out[b].item()).to(x.device)
                bg = -fg + 1
                filled.append(torch.stack([bg, fg], dim=0))  # [2, H, W, D]
            filled = torch.stack(filled, dim=0)     # [B, 2, H, W, D]
            print("cls_out:", cls_out)
            return filled
        return
    
if __name__ == "__main__":
    basepath = r"C:\Users\stoughth\mama-mia-challenge\phase1_submission\Dataset102_BreastTumor\nnUNetTrainer__nnUNetPlans__3d_fullres"
    modelPath = rf"{basepath}\fold_1\checkpoint_final.pth"
    plansPath = rf"{basepath}\plans.json"
    datasetPath = rf"{basepath}\dataset.json"
    model = myUNet(modelPath, plansPath, datasetPath)
    