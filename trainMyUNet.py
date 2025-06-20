import torch
import json
import os
# import SimpleITK as sitk
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

from MAMAMIA.nnUNet.nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from MAMAMIA.nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from MAMAMIA.nnUNet.nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder

from myUNet import myUNet

writer = None

def saveModel(trainer: nnUNetTrainer, filename: str):

    checkpoint = {
        'network_weights': trainer.network.state_dict(),
        'optimizer_state': trainer.optimizer.state_dict(),
        'grad_scaler_state': trainer.grad_scaler.state_dict() if trainer.grad_scaler is not None else None,
        'logging': trainer.logger.get_checkpoint(),
        '_best_ema': trainer._best_ema,
        'current_epoch': trainer.current_epoch + 1,
        'init_args': trainer.my_init_kwargs,
        'trainer_name': trainer.__class__.__name__,
        'inference_allowed_mirroring_axes': trainer.inference_allowed_mirroring_axes,
    }
    torch.save(checkpoint, filename)

def setupTrainer(plansJSONPath: str,
                 config: str, 
                 fold: int, 
                 datasetJSONPath: str, 
                 device: torch.device,
                 pretrainedModelPath: str = None,
                 tag: str = "") -> nnUNetTrainer:
    
    with open(plansJSONPath, 'r', encoding="utf-8") as fp:
        plans = json.load(fp)
        # plans = resolve_modules_in_dict(plans)
        kwargs = plans["configurations"][config]["architecture"]["arch_kwargs"]
        expectedChannels = kwargs["features_per_stage"]
        expectedStride = [x[0] for x in kwargs["strides"]]      # these are saved per-channel, but I'm putting all channels through at once

    with open(datasetJSONPath, 'r', encoding="utf-8") as fp:
        datasetInfo: dict = json.load(fp)
        nInChannels = len(datasetInfo["channel_names"])

    trainer = nnUNetTrainer(plans, config, fold, datasetInfo, device, tag)
    trainer.num_iterations_per_epoch = 200
    trainer.num_val_iterations_per_epoch = 50
    trainer.num_epochs = 1000
    trainer.initialize()

    model = myUNet(trainer.network, nInChannels, expectedChannels, expectedStride, pretrainedModelPath).to(device)
    trainer.network = model

    # change optimizer and scheduler
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    num_training_steps = trainer.num_epochs           # scheduler steps every epoch, not every batch
    num_warmup_steps = int(0.2 * num_training_steps)  # 20% warmup

    trainer.lr_scheduler = get_cosine_schedule_with_warmup(
        trainer.optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps,
        num_cycles=5
    )

    trainer.disable_checkpointing = True    # we will do this manually

    return trainer

def train(trainer: nnUNetTrainer, continue_training: bool = False):
    if continue_training:
        trainer.load_checkpoint(os.path.join(trainer.output_folder, 'checkpoint_latest.pth'))
        print(f"Continuing training from epoch {trainer.current_epoch}")
    else:
        trainer.current_epoch = 0
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

        if epoch % trainer.save_every == 0 and trainer.current_epoch != (trainer.num_epochs - 1):
            saveModel(trainer, os.path.join(trainer.output_folder, 'checkpoint_latest_myUNet.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if trainer._best_ema is None or trainer.logger.my_fantastic_logging['ema_fg_dice'][-1] > trainer._best_ema:
            saveModel(trainer, os.path.join(trainer.output_folder, 'checkpoint_best_myUNet.pth'))
            saveModel(trainer, os.path.join(trainer.output_folder, 'checkpoint_latest_myUNet.pth'))

        trainer.on_epoch_end()
    
    saveModel(trainer, os.path.join(trainer.output_folder, "checkpoint_final_myUNet.pth"))
    trainer.on_train_end()

def inference(trainer: nnUNetTrainer, state_dict_path: str, outputPath: str = "./outputs"):
    trainer.set_deep_supervision_enabled(False)
    state_dict = torch.load(state_dict_path, map_location=trainer.device, weights_only=False)['network_weights']
    trainer.network.load_state_dict(state_dict)
    trainer.network.eval()

    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=trainer.device, verbose=False,
                                verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(trainer.network, trainer.plans_manager, trainer.configuration_manager,
                                    [state_dict], trainer.dataset_json, trainer.__class__.__name__,
                                    trainer.inference_allowed_mirroring_axes)
    inputFolder = os.path.join(os.environ["nnUNet_raw"], datasetName, "imagesTs")

    os.makedirs(outputPath, exist_ok=True)
    ret = predictor.predict_from_files(
        inputFolder,
        outputPath
    )

    metrics = compute_metrics_on_folder(os.path.join(trainer.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                        outputPath,
                                        os.path.join(outputPath, 'summary.json'),
                                        trainer.plans_manager.image_reader_writer_class(),
                                        trainer.dataset_json["file_ending"],
                                        trainer.label_manager.foreground_regions if trainer.label_manager.has_regions else
                                        trainer.label_manager.foreground_labels,
                                        trainer.label_manager.ignore_label)
    
    trainer.print_to_log_file("Validation complete", also_print_to_console=True)
    trainer.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                              also_print_to_console=True)

# def postProcess(segmentationPath: str):
#     from scipy.ndimage import label, binary_opening, binary_closing, binary_fill_holes
#     from numpy import ndarray, bincount
#     def getPrimaryTumor(seg_array: ndarray):
#         labeled_mask, _ = label(seg_array)
#         sizes = bincount(labeled_mask.ravel())
#         sizes[0] = 0  # background
#         largest_label = sizes.argmax()
#         primary_tumor = (labeled_mask == largest_label).astype(int)
#         return primary_tumor

#     for seg in os.listdir(segmentationPath):
#         segPath = os.path.join(segmentationPath, seg)
#         segImg = sitk.ReadImage(segPath)
#         primaryTumorArr = getPrimaryTumor(sitk.GetArrayFromImage(segImg))
#         primaryTumorImg = sitk.GetImageFromArray(primaryTumorArr)
#         primaryTumorImg.CopyInformation(segImg)
#         sitk.WriteImage(primaryTumorImg, segPath)

if __name__ == "__main__":
    writer = SummaryWriter()
    datasetName = "Dataset104_cropped_3ch_breast"
    basepath = rf"{os.environ["nnUNet_preprocessed"]}/{datasetName}"
    pretrainedModelPath = "nnunet_pretrained_weights_64_final.pth"
    plansPath = rf"{basepath}/nnUNetPlans.json"
    datasetPath = rf"{basepath}/dataset.json"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")    # Joe is using the GPU rn :p
    print(f"Using device: {device}")
    fold = 4
    tag = "_transformer_128_skips"
    trainer = setupTrainer(plansPath, 
                           "3d_fullres", 
                           fold, 
                           datasetPath, 
                           device, 
                           pretrainedModelPath, 
                           tag=tag)
    state_dict_path = rf"{os.environ["nnUNet_results"]}/Dataset104_cropped_3ch_breast/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold}{tag}/checkpoint_best_myUNet.pth"
    # train(trainer)
    inference(trainer, state_dict_path, outputPath=rf"{os.environ["nnUNet_results"]}/Dataset104_cropped_3ch_breast/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold}{tag}/pred_segmentations_cropped")