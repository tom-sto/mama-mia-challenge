import torch
from myTransformer import MyTransformer, ClassifierHead

class myUNet(torch.nn.Module):
    def __init__(self, 
                 inChannels: int,
                 expectedChannels: list[int], 
                 expectedStride: list[int],
                 n_heads: int,
                 n_layers: int,
                 ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p_split = 4
        self.encoder = MyTransformer(expectedChannels, expectedStride, inChannels, p_split, transformer_depth=n_layers, num_heads=n_heads)
        self.classifier = ClassifierHead(dim=expectedChannels[-1], metadata_d=n_heads)

        self.ret = "logits"

    def forward(self, x: torch.Tensor, metadata: list = None):
        B = x.shape[0]
        imgShape = x.shape[2:]
        transformer_tokens = self.encoder(x, metadata)
        # print("transformer out shape:", transformer_tokens.shape)
        clsOut = self.classifier(transformer_tokens)
        # print("cls Out shape:", clsOut.shape)

        if self.ret == 'logits':
            return clsOut
        
        if self.ret == 'probability':
            return torch.sigmoid(clsOut)

        if self.ret == 'binary':
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
    