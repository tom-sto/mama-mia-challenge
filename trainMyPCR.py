import torch
import numpy as np
import json
import os
# import SimpleITK as sitk
from torch.utils.tensorboard import SummaryWriter

from MAMAMIA.nnUNet.nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from MAMAMIA.nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from MAMAMIA.nnUNet.nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from MAMAMIA.nnUNet.nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from MAMAMIA.nnUNet.nnunetv2.utilities.collate_outputs import collate_outputs

from myPCR import PCRPredictor

writer = None
class PCRLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        '''
        Expect x to be pCR prediction: B x 1
        Expect y to be pCR classification: B x 1 x patch_size x patch_size x patch_size
            All the values in this tensor should be the same
        Return signed difference of x and the value that y is filled with
        '''
        y_value = y.view(y.size(0), -1)[:, 0]  # shape (B,)
        x = x.view(-1)  # shape (B,)
        diff = x - y_value  # shape (B,)
        # To avoid cancellation, use absolute value before reduction
        return torch.mean(torch.abs(diff))

def validation_step(trainer: nnUNetTrainer, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        keys = batch['keys']

        metadata = trainer.get_metadata(keys)

        data = data.to(trainer.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(trainer.device, non_blocking=True) for i in target]
        else:
            target = target.to(trainer.device, non_blocking=True)

        with torch.autocast(trainer.device.type, enabled=True):
            output = trainer.network(data, metadata)
            del data
            l = trainer.loss(output, target)

        return {'loss': l.detach().cpu().numpy()}

def on_validation_epoch_end(trainer: nnUNetTrainer, val_outputs: list[dict]):
        outputs_collated = collate_outputs(val_outputs)
        loss_here = np.mean(outputs_collated['loss'])

        trainer.logger.log('mean_fg_dice', 0., trainer.current_epoch)
        trainer.logger.log('dice_per_class_or_region', [0.], trainer.current_epoch)
        trainer.logger.log('val_losses', loss_here, trainer.current_epoch)

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
                 tag: str = "") -> nnUNetTrainer:
    
    with open(plansJSONPath, 'r', encoding="utf-8") as fp:
        plans = json.load(fp)
        
    with open(datasetJSONPath, 'r', encoding="utf-8") as fp:
        datasetInfo: dict = json.load(fp)

    trainer = nnUNetTrainer(plans, config, fold, datasetInfo, device, tag)
    trainer.enable_deep_supervision = False
    trainer.num_iterations_per_epoch = 900
    trainer.num_val_iterations_per_epoch = 200
    trainer.num_epochs = 300
    trainer.initial_lr = 1e-3
    trainer.initialize()

    trainer.set_deep_supervision_enabled = lambda _: _      # do nothing with this!

    model = PCRPredictor(device)
    trainer.network = model.to(device)

    # change optimizer and scheduler
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.initial_lr, weight_decay=1e-4)
    trainer.lr_scheduler = PolyLRScheduler(trainer.optimizer, trainer.initial_lr, trainer.num_epochs)

    trainer.loss = PCRLoss()
    trainer.disable_checkpointing = True

    return trainer

def train(trainer: nnUNetTrainer, continue_training: bool = False):
    if continue_training:
        trainer.load_checkpoint(os.path.join(trainer.output_folder, 'checkpoint_latest.pth'))
        print(f"Continuing training from epoch {trainer.current_epoch}")
    else:
        trainer.current_epoch = 0

    best_val_loss = 10000
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
                val_outputs.append(validation_step(trainer, next(trainer.dataloader_val)))
            on_validation_epoch_end(trainer, val_outputs)

        if epoch % trainer.save_every == 0 and trainer.current_epoch != (trainer.num_epochs - 1):
            saveModel(trainer, os.path.join(trainer.output_folder, 'checkpoint_latest_myPCR.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if trainer.logger.my_fantastic_logging['val_losses'][-1] > best_val_loss:
            saveModel(trainer, os.path.join(trainer.output_folder, 'checkpoint_best_myPCR.pth'))
            saveModel(trainer, os.path.join(trainer.output_folder, 'checkpoint_latest_myPCR.pth'))
            best_val_loss = trainer.logger.my_fantastic_logging['val_losses'][-1]
            trainer.print_to_log_file(f"Yayy! New best val loss {best_val_loss}")

        trainer.on_epoch_end()
    
    saveModel(trainer, os.path.join(trainer.output_folder, "checkpoint_final_myPCR.pth"))
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

if __name__ == "__main__":
    writer = SummaryWriter()
    datasetName = "Dataset200_cropped_4ch_breast_and_seg"
    basepath = rf"{os.environ["nnUNet_preprocessed"]}/{datasetName}"
    plansPath = rf"{basepath}/nnUNetPlans.json"
    datasetPath = rf"{basepath}/dataset.json"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    fold = 4
    tag = "_pcr_more_layers"
    trainer = setupTrainer(plansPath, 
                           "3d_fullres", 
                           fold, 
                           datasetPath, 
                           device,  
                           tag=tag)
    state_dict_path = rf"{os.environ["nnUNet_results"]}/{datasetName}/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold}{tag}/checkpoint_best_myUNet.pth"
    train(trainer)
    # inference(trainer, state_dict_path, outputPath=rf"{os.environ["nnUNet_results"]}/Dataset104_cropped_3ch_breast/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold}{tag}/pred_segmentations_cropped")