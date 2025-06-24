import torch
import json
import os
import SimpleITK as sitk
from torch.utils.tensorboard import SummaryWriter

from MAMAMIA.nnUNet.nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from MAMAMIA.nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from MAMAMIA.nnUNet.nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from MAMAMIA.nnUNet.nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder

from myRefiner import MyRefiner

writer = None

def setupTrainer(plansJSONPath: str,
                 config: str, 
                 fold: int, 
                 datasetJSONPath: str, 
                 device: torch.device,
                 tag=""):
    
    with open(plansJSONPath, 'r', encoding="utf-8") as fp:
        plans = json.load(fp)

    with open(datasetJSONPath, 'r', encoding="utf-8") as fp:
        datasetInfo: dict = json.load(fp)

    trainer = nnUNetTrainer(plans, config, fold, datasetInfo, device, tag)
    trainer.weight_bd = 0
    trainer.num_iterations_per_epoch = 1000
    trainer.num_val_iterations_per_epoch = 400
    trainer.num_epochs = 800
    trainer.enable_deep_supervision = False
    trainer.initial_lr = 5e-5
    trainer.initialize()

    model = MyRefiner().to(device)
    trainer.network = model
    trainer.oversample_foreground_percent = 0.8

    # change optimizer and scheduler
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.initial_lr, weight_decay=1e-5)
    trainer.lr_scheduler = PolyLRScheduler(trainer.optimizer, trainer.initial_lr, trainer.num_epochs)

    # print loss fn
    print(f"Loss function: {trainer.loss}")

    # trainer.disable_checkpointing = True    # we will do this manually

    return trainer

def train(trainer: nnUNetTrainer, resume: bool = False, checkpoint: str = None):
    if resume:
        assert checkpoint is not None, "Cannot resume training if no checkpoint is given!"
        trainer.load_checkpoint(checkpoint)

    trainer.on_train_start()

    for epoch in range(trainer.current_epoch, trainer.num_epochs):
        trainer.on_epoch_start()

        trainer.on_train_epoch_start()
        train_outputs = []
        for batch_id in range(trainer.num_iterations_per_epoch):
            train_outputs.append(trainer.train_step(next(trainer.dataloader_train)))
        trainer.on_train_epoch_end(train_outputs)

        total_norm = 0
        for p in trainer.network.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)  # L2 norm
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        writer.add_scalar("Gradient Norm", total_norm, epoch)

        with torch.no_grad():
            trainer.on_validation_epoch_start()
            val_outputs = []
            for batch_id in range(trainer.num_val_iterations_per_epoch):
                val_outputs.append(trainer.validation_step(next(trainer.dataloader_val)))
            trainer.on_validation_epoch_end(val_outputs)
        
        trainer.on_epoch_end()
    
    trainer.save_checkpoint(os.path.join(trainer.output_folder, "checkpoint_final_myRefiner.pth"))
    trainer.on_train_end()

def inference(trainer: nnUNetTrainer, state_dict_path: str, inputFolder: str, outputPath: str = "./outputs"):
    trainer.set_deep_supervision_enabled(False)
    state_dict = trainer.load_checkpoint(state_dict_path)
    trainer.network.eval()

    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=trainer.device, verbose=False,
                                verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(trainer.network, trainer.plans_manager, trainer.configuration_manager,
                                    [state_dict], trainer.dataset_json, trainer.__class__.__name__,
                                    trainer.inference_allowed_mirroring_axes)
    

    os.makedirs(outputPath, exist_ok=True)
    ret = predictor.predict_from_files(
        inputFolder,
        outputPath
    )

    forCorrupted = 'Dataset666_corrupted_preds' in outputPath

    from score_task1 import doScoring
    doScoring(os.path.dirname(outputPath), forCorrupted)

def postProcess(segmentationPath: str):
    from scipy.ndimage import label, binary_opening, binary_closing, binary_fill_holes
    from numpy import ndarray, bincount
    def getPrimaryTumor(seg_array: ndarray):
        labeled_mask, _ = label(seg_array)
        sizes = bincount(labeled_mask.ravel())
        sizes[0] = 0  # background
        largest_label = sizes.argmax()
        primary_tumor = (labeled_mask == largest_label).astype(int)
        return primary_tumor

    for seg in os.listdir(segmentationPath):
        segPath = os.path.join(segmentationPath, seg)
        segImg = sitk.ReadImage(segPath)
        primaryTumorArr = getPrimaryTumor(sitk.GetArrayFromImage(segImg))
        primaryTumorImg = sitk.GetImageFromArray(primaryTumorArr)
        primaryTumorImg.CopyInformation(segImg)
        sitk.WriteImage(primaryTumorImg, segPath)

if __name__ == "__main__":
    writer = SummaryWriter()
    datasetName = "Dataset666_corrupted_preds"
    basepath = rf"{os.environ["nnUNet_preprocessed"]}\{datasetName}"
    plansPath = rf"{basepath}\nnUNetPlans.json"
    datasetPath = rf"{basepath}\dataset.json"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    fold = 4
    tag = "_with_mri_64"
    trainer = setupTrainer(plansPath, "3d_fullres", fold, datasetPath, device, tag)
    state_dict_path = os.path.join(trainer.output_folder, "checkpoint_final.pth")
    train(trainer)
    inputFolder = os.path.join(os.environ["nnUNet_raw"], datasetName, "imagesTs")
    outputFolder = os.path.join(os.environ["nnUNet_results"], 
                                datasetName, 
                                "nnUNetTrainer__nnUNetPlans__3d_fullres", 
                                f"fold_{fold}{tag}", 
                                "results",
                                "pred_segmentations_cropped")
    inference(trainer, state_dict_path, inputFolder, outputFolder)
