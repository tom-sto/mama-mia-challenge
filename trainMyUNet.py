import torch
import json
import os
import math
# import SimpleITK as sitk
# from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
from MAMAMIA.nnUNet.nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from MAMAMIA.nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from MAMAMIA.nnUNet.nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from myUNet import myUNet

writer = None

class WarmupCosineAnnealingWithRestarts(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, cycle_steps, cycle_mult=1.0, max_lr=1e-3, min_lr=1e-5, damping=1.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.damping = damping

        self.cur_cycle = 0
        self.cycle_progress = 0
        self.next_cycle_step = warmup_steps + cycle_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        # Warmup phase
        if step < self.warmup_steps:
            warmup_lr = self.min_lr + (self.max_lr - self.min_lr) * step / self.warmup_steps
            return [warmup_lr for _ in self.base_lrs]

        # Update cycle
        if step >= self.next_cycle_step:
            self.cur_cycle += 1
            self.cycle_steps = int(self.cycle_steps * self.cycle_mult)
            self.next_cycle_step = step + self.cycle_steps
            self.cycle_progress = 0
        else:
            self.cycle_progress = step - (self.next_cycle_step - self.cycle_steps)

        cycle_ratio = self.cycle_progress / self.cycle_steps
        damped_max_lr = self.max_lr * (self.damping ** self.cur_cycle)
        cosine_lr = self.min_lr + 0.5 * (damped_max_lr - self.min_lr) * (1 + math.cos(math.pi * cycle_ratio))

        return [cosine_lr for _ in self.base_lrs]


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
    trainer.initial_lr = 1e-5
    trainer.initialize()

    model = myUNet(trainer.network, nInChannels, expectedChannels, expectedStride, pretrainedModelPath).to(device)
    trainer.network = model

    # change optimizer and scheduler
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.initial_lr, weight_decay=1e-4)

    num_training_steps = trainer.num_epochs           # scheduler steps every epoch, not every batch
    num_warmup_steps = round(0.2 * num_training_steps)  # 20% warmup
    num_cycle_steps = round(0.1 * num_training_steps) + 1

    trainer.lr_scheduler = WarmupCosineAnnealingWithRestarts(
        trainer.optimizer,
        warmup_steps=num_warmup_steps,
        cycle_steps=num_cycle_steps,
        max_lr=trainer.initial_lr,
        min_lr=1e-10,
        damping=0.85
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
    ret = predictor.predict_from_files_sequential(
        inputFolder,
        outputPath
    )

    from score_task1 import doUncropping, generate_scores
    doUncropping(os.path.dirname(outputPath))
    generate_scores(os.path.dirname(outputPath))

if __name__ == "__main__":
    writer = SummaryWriter()
    datasetName = "Dataset104_cropped_3ch_breast"
    basepath = rf"{os.environ["nnUNet_preprocessed"]}/{datasetName}"
    pretrainedModelPath = "nnunet_pretrained_weights_64_best.pth"
    plansPath = rf"{basepath}/nnUNetPlans.json"
    datasetPath = rf"{basepath}/dataset.json"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")    # Joe is using the GPU rn :p
    print(f"Using device: {device}")
    fold = 4
    tag = "_transformer_128_skips_fair_no_transpose"
    trainer = setupTrainer(plansPath, 
                           "3d_fullres", 
                           fold, 
                           datasetPath, 
                           device, 
                           pretrainedModelPath, 
                           tag=tag)
    state_dict_path = rf"{os.environ["nnUNet_results"]}/Dataset104_cropped_3ch_breast/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold}{tag}/checkpoint_best_myUNet.pth"
    
    # lr = []
    # for epoch in range(trainer.num_epochs):
    #     trainer.lr_scheduler.step()
    #     lr.append(trainer.lr_scheduler.get_last_lr()[0])
    # import matplotlib.pyplot as plt
    # plt.plot(lr)
    # plt.title("Learning Rate Schedule")
    # plt.xlabel("Epoch")
    # plt.ylabel("Learning Rate")
    # plt.savefig("lr_schedule.png")
    train(trainer)
    inference(trainer, state_dict_path, outputPath=rf"{os.environ["nnUNet_results"]}/Dataset104_cropped_3ch_breast/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold}{tag}/outputs/pred_segmentations_cropped")
