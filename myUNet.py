import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from myTransformer import MyTransformer, ClassifierHead
# from stTransformer import TransformerST
from TransformerTwo import TransformerST, ClassifierHead

class MyUNet(torch.nn.Module):
    def __init__(self, 
                 expectedPatchSize: int,
                 expectedChannels: list[int] = [32, 64, 128, 256, 320, 320], 
                 expectedStride: list[int] = [1, 2, 2, 2, 2, 2],
                 pretrainedModelPath: str = None,
                 p_split: int = 4,
                 n_heads: int = 8):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # build UNet from default args (see plans.json from baseline nnUNet run)
        unet = PlainConvUNet(input_channels=1, 
                             n_stages=6, 
                             features_per_stage=expectedChannels, 
                             conv_op=nn.Conv3d, 
                             kernel_sizes=3, 
                             strides=expectedStride, 
                             n_conv_per_stage=(2, 2, 2, 2, 2, 2), 
                             num_classes=2,                                 # Binary segmentation!
                             n_conv_per_stage_decoder=(2, 2, 2, 2, 2),
                             conv_bias=True, 
                             norm_op=nn.InstanceNorm3d, 
                             norm_op_kwargs={"eps": 1e-05, "affine": True}, 
                             dropout_op=None, 
                             dropout_op_kwargs=None, 
                             nonlin=nn.ReLU, 
                             nonlin_kwargs={"inplace": True},
                             deep_supervision=True)

        # only care about the decoder here
        # but this decoder expects to see the encoder on initializing, which is why we use the full model to start
        decoder = unet.decoder
        
        if pretrainedModelPath is not None:
            
            stateDict: dict = torch.load(pretrainedModelPath, map_location='cpu', weights_only=False)['network_weights']

            # Load only decoder weights
            decoderStateDict = {k.replace("decoder.", ""): v for k, v in stateDict.items() if "decoder" in k}
            decoder.load_state_dict(decoderStateDict, strict=False)

        # as long as we can provide the correct-shaped skip connections, we're good to do whatever we want with the encoder after init
        self.encoder = TransformerST(expectedPatchSize, expectedChannels, expectedStride, num_heads=n_heads, transformer_num_layers=4, p_split=p_split)
        self.decoder = decoder
        self.classifier = ClassifierHead(dim=expectedChannels[-1])

        self.ret = "all"

    def forward(self, x: torch.Tensor, patientData: list = None):
        features, skips, cls_token = self.encoder(x, patientData)
        segOut = self.decoder(skips)

        # print("Features:", features.shape)
        # print("cls token:", cls_token.shape)
        if self.ret == "seg":
            return None, segOut, None
        elif self.ret == "segOnly":
            return segOut

        clsOut = self.classifier(cls_token)

        if self.ret == "all":
            return features, segOut, clsOut
        elif self.ret == "prob":
            return None, None, torch.sigmoid(clsOut)
        elif self.ret == "probOnly":
            return torch.sigmoid(clsOut)
        return
    
if __name__ == "__main__":
    basepath = r"C:\Users\stoughth\mama-mia-challenge\phase1_submission\Dataset102_BreastTumor\nnUNetTrainer__nnUNetPlans__3d_fullres"
    modelPath = rf"{basepath}\fold_1\checkpoint_final.pth"
    plansPath = rf"{basepath}\plans.json"
    datasetPath = rf"{basepath}\dataset.json"
    model = MyUNet(modelPath, plansPath, datasetPath)
    