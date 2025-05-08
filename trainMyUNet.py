import torch
import json
import os

from MAMAMIA.nnUNet.nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from myUNet import myUNet

def setupTrainer(plansJSONPath: str,
                 config: str, 
                 fold: int, 
                 datasetJSONPath: str, 
                 device: torch.device,
                 pretrainedModelPath: str = None):
    with open(plansJSONPath, 'r') as fp:
        plans = json.load(fp)
        # plans = resolve_modules_in_dict(plans)
    
    with open(datasetJSONPath, 'r') as fp:
        datasetInfo: dict = json.load(fp)
    
    trainer = nnUNetTrainer(plans, config, fold, datasetInfo, device)
    trainer.initialize()

    kwargs = plans["configurations"][config]["architecture"]["arch_kwargs"]

    expectedChannels = kwargs["features_per_stage"]
    expectedStride = [x[0] for x in kwargs["strides"]]
    model = myUNet(trainer.network, expectedChannels, expectedStride, pretrainedModelPath).to(device)
    trainer.network = model

    return trainer

def train(trainer: nnUNetTrainer):
    trainer.run_training()

if __name__ == "__main__":
    basepath = rf"{os.environ["nnUNet_preprocessed"]}\Dataset103_cropped_breast"
    # modelPath = rf"{basepath}\fold_1\checkpoint_final.pth"
    modelPath = None
    plansPath = rf"{basepath}\nnUNetPlans_64patch.json"
    datasetPath = rf"{basepath}\dataset.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    trainer = setupTrainer(plansPath, "3d_fullres", 4, datasetPath, device, modelPath)
    train(trainer)

    