import torch
import os
import math
# import SimpleITK as sitk
# from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
from MAMAMIA.nnUNet.nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from MAMAMIA.nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from torchjd.aggregation import UPGrad
from myUNet import MyUNet
from CustomLoader import GetDataloaders

writer = None

class PCRLoss(torch.nn.Module):
    def __init__(self):
        """
        alpha and beta are weighting terms in case you want to combine BCE with another loss
        for example: total_loss = alpha * BCE + beta * focal or another auxiliary term.
        """
        super(PCRLoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.4))  # this data set sees 70% negative, 30% positive

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: (B,) or (B, 1) raw model outputs (before sigmoid)
            targets: list of ints (0 or 1), or a torch.Tensor of shape (B,)
        """
        # Convert list to tensor if needed
        if isinstance(targets, list):
            targets = torch.tensor(targets, device=logits.device)

        targets = targets.float()

        # Ensure logits and targets match in shape
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1).float()  # shape (B,)

        loss = self.bce(logits, targets).float()

        return loss

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

class MyTrainer():
    def __init__(self,):
        self.num_iterations_per_epoch = 20
        self.num_val_iterations_per_epoch = 5
        self.num_epochs = 1
        self.pretrainSegmentation = self.num_epochs * 1
        self.initial_lr = 1e-5
        self.current_epoch = 0

    def setup(self,
              dataDir: str,
              device: torch.device,
              pretrainedModelPath: str = None,
              tag: str = ""):

        p_split = 4
        n_heads = 8
        patch_size = 32     

        self.model = MyUNet(patch_size, pretrainedModelPath=pretrainedModelPath, n_heads=n_heads, p_split=p_split).to(device)

        self.trDataloader, self.vlDataloader, self.tsDataloader = GetDataloaders(dataDir, device)

        # non_transformer_params = [
        #     param for name, param in self.model.encoder.named_parameters()
        #     if not name.startswith("transformer")
        # ]
        # # change optimizer and scheduler
        # self.optimizer = torch.optim.AdamW([
        #     {'params': non_transformer_params, 'lr': self.initial_lr * 20, 'weight_decay': 1e-4},
        #     {'params': self.model.encoder.transformer.parameters(), 'lr': self.initial_lr, 'weight_decay': 1e-3},
        #     {'params': self.model.decoder.parameters(), 'lr': self.initial_lr,  'weight_decay': 1e-4}
        # ])
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=trainer.initial_lr, weight_decay=1e-4)
        self.aggregator = UPGrad()

        num_training_steps = self.num_epochs           # scheduler steps every epoch, not every batch
        num_warmup_steps = round(0.2 * num_training_steps)  # 10% warmup
        num_cycle_steps = round(0.4 * num_training_steps) + 1

        self.lr_scheduler = WarmupCosineAnnealingWithRestarts(
            self.optimizer,
            warmup_steps=num_warmup_steps,
            cycle_steps=num_cycle_steps,
            max_lr=self.initial_lr,
            min_lr=1e-10,
            damping=0.5
        )

        self.cls_loss = PCRLoss()

        self.disable_checkpointing = True    # we will do this manually
        self.save_every = 5

        return self

    def train(self, continue_training: bool = False):
        if continue_training:
            # self.load_checkpoint(os.path.join(self.output_folder, 'checkpoint_latest_myUNet.pth'))
            print(f"Continuing training from epoch {self.current_epoch}")

        cls_losses = []
        pcr_percentages = []
        pcr_bal_accuracy = []
        bestAccuracy = 0.
        bestJoint = 0.
        import matplotlib.pyplot as plt
        import pdb

        for epoch in range(self.num_epochs):
            for idx, struct in enumerate(self.trDataloader):        # iterate over patient cases
                obj, patientID = struct
                # print(f"thing {idx}, {patientID}: {obj.keys()}")
                seg: dict = obj['seg']
                dmap: dict = obj['dmap']
                phase: dict = obj['phase']

                encodedChunks = []

                for i in range(len(seg)):                           # iterate over chunk indices
                    segHandle, startIndex = seg[i]
                    # print(segHandle)
                    # print(startIndex)
                    dmapHandle = dmap[i]

                    target, patchIdxs = segHandle()
                    # print(target.shape)
                    # print(patchIdxs)

                    distMap = dmapHandle()
                    # print(distMap.shape)

                    phaseData = []
                    for p in phase.keys():                          # iterate over phases
                        phaseData.append(phase[p][i]())             # run the handle

                    phaseData = torch.stack(phaseData).unsqueeze(0)
                    print(phaseData.shape)

                    encodedChunks.append(self.model.encoder.patch_embed(phaseData, ))
            
            return
            
            # fix the sizing of the plots with dual y-axes
            fig, ax1 = plt.subplots(figsize=(10, 5))

            # Plot PCR Accuracy on the left y-axis
            ax1.plot(pcr_percentages, label='PCR Accuracy', color='blue')
            ax1.plot(pcr_bal_accuracy, label='Balanced Accuracy', color='green')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('PCR Accuracy', fontsize=12, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue', labelsize=10)
            ax1.tick_params(axis='x', labelsize=10)

            # Create a second y-axis for BCE Loss on the right
            ax2 = ax1.twinx()
            ax2.plot(cls_losses, label='BCE Loss', color='red')
            ax2.set_ylabel('BCE Loss', fontsize=12, color='red')
            ax2.tick_params(axis='y', labelcolor='red', labelsize=10)

            # Add a title and adjust layout
            plt.title('PCR Prediction Accuracy and BCE Loss Over Epochs', fontsize=14)
            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes, fontsize=10)
            fig.tight_layout()  # Adjust layout to prevent clipping

            # Save the plot
            plt.savefig(os.path.join(self.output_folder, 'pcr_progress.png'))
            plt.clf()
        
        saveModel(self, os.path.join(self.output_folder, "checkpoint_final_myUNet.pth"))
        self.on_train_end()

    def inference(self: nnUNetTrainer, state_dict_path: str, outputPath: str = "./outputs", outputPathPCR: str = "./outputs_pcr"):
        self.set_deep_supervision_enabled(False)
        state_dict = torch.load(state_dict_path, map_location=self.device, weights_only=False)['network_weights']
        self.network.load_state_dict(state_dict)
        self.network.eval()
        self.network.ret = "seg"

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager,
                                        [state_dict], self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)
        inputFolder = os.path.join(os.environ["nnUNet_raw"], datasetName, "imagesTs")       # THESE IMAGES ARE CROPPED

        os.makedirs(outputPath, exist_ok=True)
        # ret = predictor.predict_from_files(
        #     inputFolder,
        #     outputPath
        # )

        # from score_task1 import doScoring
        # doScoring(os.path.dirname(outputPath))
        import pdb, pandas as pd, SimpleITK as sitk
        # pdb.set_trace()
        
        # do pCR inference
        self.network.ret = "probability"
        out_df = pd.DataFrame(columns=['patient_id', 'pcr_prob'])
        patientIDS_seen = []
        for imgName in os.listdir(inputFolder):
            patientID = imgName.split('.')[0][:-5]
            if patientID in patientIDS_seen:
                continue
            patientIDS_seen.append(patientID)
            print(f"Predicting for patient {patientID}")
            
            arrs = []
            for i in range(3):
                imgPath = os.path.join(inputFolder, f"{patientID}_000{i}.nii.gz")

                arr = sitk.GetArrayFromImage(sitk.ReadImage(imgPath))
                arr = torch.from_numpy(arr).unsqueeze(0)
                arrs.append(arr)

            with torch.autocast(self.device.type):
                arr = torch.cat(arrs)
                print(f"Arr.shape = {arr.shape}")
                assert len(arr.shape) == 4, f'Expected shape (C, X, Y, Z), got shape {arr.shape}'
                arr = pad_to_patch_compatible_size(arr, 32)
                print(f"arr shape after padding: {arr.shape}")
                arr = arr.unsqueeze(0).to(self.device).float()
                p: torch.Tensor = self.network(arr)

            data = pd.DataFrame({
                'patient_id': [patientID.upper()],
                'pcr_prob': [p.item()],
            })
            out_df = pd.concat([out_df, data], ignore_index=True)

        pred_path = os.path.join(os.path.dirname(outputPathPCR), 'pcr_predictions.csv')
        print(f"Saving to {pred_path}")
        out_df.to_csv(pred_path, index=False)

        from predictPCR import scorePCR
        scorePCR(pred_path)
        from MAMAMIA.src.challenge.scoring_task2 import doScoring
        doScoring(os.path.dirname(pred_path))
    

if __name__ == "__main__":
    writer = SummaryWriter()
    datasetName = "Dataset106_cropped_Xch_breast_no_norm"
    dataDir = rf"E:\MAMA-MIA\my_preprocessed_data\{datasetName}"
    # pretrainedModelPath = "pretrained_weights/nnunet_pretrained_weights_64_best.pth"
    pretrainedModelPath = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tag = "_transformerST_joint"
    trainer = MyTrainer()
    
    output_folder = rf"{os.environ["nnUNet_results"]}/run{tag}"
    state_dict_path = rf"{output_folder}/checkpoint_best_joint.pth"
    
    trainer.setup(dataDir, device, pretrainedModelPath, tag)
    trainer.train()
    # trainer.inference(state_dict_path,
    #                   outputPath=rf"{output_folder}/outputs/pred_segmentations_cropped",
    #                   outputPathPCR=rf"{output_folder}/outputs/pred_PCR_cropped")
