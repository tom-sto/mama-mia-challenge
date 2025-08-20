import torch
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
# from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torchjd.aggregation import UPGrad
from torchjd import mtl_backward
from MyUNet import MyUNet
from Losses import PCRLoss, SegLoss, Dice
from Schedulers import WarmupCosineAnnealingWithRestarts
from CustomLoader import GetDataloaders
from DataProcessing import reconstructImageFromPatches
from helpers import PATCH_SIZE, Mean, FormatSeconds
from time import time

class MyTrainer():
    def __init__(self, nEpochs: int, modelName: str = "", tag: str = "", joint: bool = True):
        self.nEpochs = nEpochs
        self.joint = joint
        self.pretrainSegmentation = self.nEpochs * 0.5
        self.peakLR = 1e-5
        self.minLR = 1e-10
        self.currentEpoch = 0

        self.clsLosses = []
        self.PCRPercentages = []
        self.PCRBalancedAccs = []

        self.outputFolder = f"./transformerResults/{modelName}"
        os.makedirs(self.outputFolder, exist_ok=True)

        self.writer = SummaryWriter(os.path.join(self.outputFolder, f"log{tag}"))
        self.logGradients = True

    def setup(self,
              dataDir: str,
              device: torch.device,
              pretrainedDecoderPath: str = None,
              useSkips: bool = True,
              bottleneck: str = "MyTransformer",
              test: bool = False):

        nHeads = 8
        patchSize = 32

        self.model = MyUNet(patchSize,
                            pretrainedDecoderPath=pretrainedDecoderPath,
                            nHeads=nHeads,
                            useSkips=useSkips,
                            joint=self.joint,
                            bottleneck=bottleneck).to(device)
        
        self.trDataloader, self.vlDataloader, self.tsDataloader = GetDataloaders(dataDir, device, test=test)

        # change optimizer and scheduler
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.encoder.parameters(), 'lr': self.minLR * 20, 'weight_decay': 1e-4},
            {'params': self.model.bottleneck.parameters(), 'lr': self.minLR, 'weight_decay': 1e-4},
            {'params': self.model.decoder.parameters(), 'lr': self.minLR * 10,  'weight_decay': 1e-4},
            {'params': self.model.classifier.parameters(), 'lr': self.minLR * 50,  'weight_decay': 1e-4}
        ])
        # self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.minLR, weight_decay=1e-4)
        # use much smaller initial scale since our early losses will be close to 1
        self.gradScaler = torch.GradScaler(device.type, init_scale=2**8)
        self.aggregator = UPGrad()

        nWarmupSteps = round(0.2 * self.nEpochs)  # 20% warmup
        nCycleSteps = round(0.4 * self.nEpochs) + 1   # do 2 cycles

        self.LRScheduler = WarmupCosineAnnealingWithRestarts(
            self.optimizer,
            warmup_steps=nWarmupSteps,
            cycle_steps=nCycleSteps,
            maxLR=self.peakLR,
            minLR=self.minLR,
            damping=0.5
        )

        self.PCRloss = PCRLoss()
        self.SegLoss = SegLoss()

        return self

    def train(self, continue_training: bool = False):
        if continue_training:
            # self.load_checkpoint(os.path.join(self.outputFolder, 'checkpoint_latest_myUNet.pth'))
            print(f"Continuing training from epoch {self.currentEpoch}")

        bestSeg = 10.
        bestPCR = 10.
        bestJoint = 10.

        print("Starting training!")
        start = time()
        for epoch in range(self.nEpochs):
            self.currentEpoch = epoch
            print(f"Epoch {self.currentEpoch}:")

            # =========================================
            #               TRAINING LOOP
            # =========================================

            segLossesThisEpoch = []
            pcrLossesThisEpoch = []

            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("LR", lr, self.currentEpoch)

            startEpoch = time()
            for idx, struct in enumerate(self.trDataloader):        # iterate over patient cases
                obj, patientIDs = struct

                if type(patientIDs) == str:
                    patientIDs = [patientIDs]
                # print(f"thing {idx}, {patientIDs}: {obj.keys()}")
                seg: dict = obj['seg']
                dmap: dict = obj['dmap']
                phase: dict = obj['phase']
                # pcr: dict = obj['pcr']

                for i in range(len(seg)):                           # iterate over chunk indices
                    target, patchIdxs = seg[i]()
                    distMap = dmap[i]()
                    # pcrVal = pcr[i]()

                    target: torch.Tensor = target.unsqueeze(1)      # add singleton channel dim
                    distMap: torch.Tensor = distMap.unsqueeze(1)    # add singleton channel dim

                    # run the phase handles
                    phaseData = [phase[p][i]() for p in phase.keys()]
                    phaseData = torch.stack(phaseData).unsqueeze(0)

                    with torch.autocast(self.model.device.type):
                        segOut, sharedFeatures, pcrOut = self.model(phaseData, patientIDs, patchIdxs)

                        segLoss: torch.Tensor = self.SegLoss(segOut, target, distMap)
                        pcrLoss = None
                        if self.currentEpoch > self.pretrainSegmentation and pcrOut and self.joint:
                            pcrLoss: torch.Tensor = self.PCRloss(pcrOut, pcrVal)
                            del pcrOut
                
                    segLossesThisEpoch.append(segLoss.item())
                    self.optimizer.zero_grad()
                    if self.joint and pcrLoss:
                        assert sharedFeatures is not None, "Cannot do joint backward without shared features!"
                        pcrLossesThisEpoch.append(pcrLoss.item())

                        scaledLosses = self.gradScaler.scale([segLoss, pcrLoss])
                        mtl_backward(losses=scaledLosses, 
                                     features=sharedFeatures, 
                                     aggregator=self.aggregator,
                                     tasks_params=[list(self.model.decoder.parameters()), 
                                                   list(self.model.classifier.parameters())],
                                     shared_params=list(self.model.encoder.parameters()) + list(self.model.bottleneck.parameters()))
                    else:
                        scaledLoss = self.gradScaler.scale(segLoss)
                        scaledLoss.backward()
                        
                    self.gradScaler.step(self.optimizer)
                    self.gradScaler.update()

                    del segOut, sharedFeatures
                    torch.cuda.empty_cache()

            self.writer.add_scalar('Seg Loss/Training', Mean(segLossesThisEpoch), self.currentEpoch)
            if len(pcrLossesThisEpoch) > 0:
                self.writer.add_scalar('PCR Loss/Training', Mean(pcrLossesThisEpoch), self.currentEpoch)
                self.writer.add_scalar('Joint Loss/Training', Mean([s + p for s, p in zip(segLossesThisEpoch, pcrLossesThisEpoch)]), self.currentEpoch)

            if self.currentEpoch % 10 == 0 and self.logGradients:
                for name, param in self.model.named_parameters():
                    if param.grad is not None and ('encoder' in name or 'bottleneck' in name or 'decoder'):
                        self.writer.add_histogram(f'Training Gradients/{name}', param.grad, self.currentEpoch)

            print(f"\tTraining loop took {FormatSeconds(time() - startEpoch)}")
            
            # =========================================
            #              VALIDATION LOOP
            # =========================================

            segLossesVal = []
            pcrLossesVal = []
            diceVal = []
            for idx, struct in enumerate(self.vlDataloader):        # iterate over patient cases
                obj, patientIDs = struct

                # we may allow batching in the future! for now just toss the one patientID into a list
                if type(patientIDs) == str:
                    patientIDs = [patientIDs]
                # print(f"thing {idx}, {patientIDs}: {obj.keys()}")

                seg: dict = obj['seg']
                dmap: dict = obj['dmap']
                phase: dict = obj['phase']
                # pcr: dict = obj['pcr']

                for i in range(len(seg)):                           # iterate over chunk indices
                    target, patchIdxs = seg[i]()
                    distMap = dmap[i]()
                    # pcrVal = pcr[i]()

                    target: torch.Tensor = target.unsqueeze(1)      # add singleton channel dim
                    distMap: torch.Tensor = distMap.unsqueeze(1)    # add singleton channel dim

                    # run the phase handles
                    phaseData = [phase[p][i]() for p in phase.keys()]
                    phaseData = torch.stack(phaseData).unsqueeze(0)

                    with torch.autocast(self.model.device.type):
                        segOut, _, pcrOut = self.model(phaseData, patientIDs, patchIdxs)

                        segLoss: torch.Tensor = self.SegLoss(segOut, target, distMap)
                        pcrLoss = None
                        if self.currentEpoch > self.pretrainSegmentation and pcrOut and self.joint:
                            pcrLoss: torch.Tensor = self.PCRloss(pcrOut, pcrVal)
                            del pcrOut

                    segOut: torch.Tensor = (segOut > 0).int()
                    segImageArr = reconstructImageFromPatches(segOut.transpose(1, 0), [patchIdxs], PATCH_SIZE)
                    targetImageArr = reconstructImageFromPatches(target.transpose(1, 0), [patchIdxs], PATCH_SIZE)
                    
                    dice = Dice(segImageArr, targetImageArr)
                    diceVal.append(dice)

                    segLossesVal.append(segLoss.item())
                    if self.joint and pcrLoss:
                        pcrLossesVal.append(pcrLoss.item())
                    # binary_preds: torch.Tensor = (torch.sigmoid(cls_out) > 0.5).bool()
                    # pcrLabels = pcrLabels.bool()
                    # correct: torch.Tensor = binary_preds == pcrLabels
                    # percentage_correct = correct.float().mean()
                    # tp_pcr = (binary_preds & pcrLabels).sum()
                    # tn_pcr = (~binary_preds & ~pcrLabels).sum()
                    # fp_pcr = (binary_preds & ~pcrLabels).sum()
                    # fn_pcr = (~binary_preds & pcrLabels).sum()
                    # sensitivity = tp_pcr / (tp_pcr + fn_pcr) if (tp_pcr + fn_pcr).item() > 0 else torch.tensor(0.)
                    # specificity = tn_pcr / (tn_pcr + fp_pcr) if (tn_pcr + fp_pcr).item() > 0 else torch.tensor(0.)
                    # balanced_accuracy = (sensitivity + specificity) / 2

                    del segOut
                    torch.cuda.empty_cache()
            
            self.LRScheduler.step()

            # MODEL CHECKPOINTING
            avgSegValLoss = Mean(segLossesVal)
            self.writer.add_scalar('Seg Loss/Validation', avgSegValLoss, self.currentEpoch)

            avgDiceVal = Mean(diceVal)
            self.writer.add_scalar('Validation Dice', avgDiceVal, self.currentEpoch)

            if avgSegValLoss < bestSeg:
                bestSeg = avgSegValLoss
                self.saveModel(f"BestSeg")
            
            if len(pcrLossesVal) > 0:
                avgPCRValLoss = Mean(pcrLossesVal)
                self.writer.add_scalar('PCR Loss/Validation', avgPCRValLoss, self.currentEpoch)

                avgJoint = Mean([s + p for s, p in zip(segLossesVal, pcrLossesVal)])
                self.writer.add_scalar('Joint Loss/Validation', avgJoint, self.currentEpoch)

                if avgPCRValLoss < bestPCR:
                    bestPCR = avgPCRValLoss
                    self.saveModel(f"BestPCR")
                
                if avgJoint < bestJoint:
                    bestJoint = avgJoint
                    self.saveModel(f"BestJoint")

            if self.currentEpoch % 5 == 0:
                self.saveModel()

            print(f"\tFull loop took {FormatSeconds(time() - startEpoch)}")
        # print(patchIdxs)

        print("Done training!")
        print(f"\tTook {FormatSeconds(time() - start)}.")
        return
        segOut = (segOut > 0.5).int()

        print(segOut.shape)
        
        reconSeg = reconstructImageFromPatches(segOut.detach().cpu(), [patchIdxs], PATCH_SIZE)
        sitk.WriteImage(sitk.GetImageFromArray(reconSeg.numpy()), "reconSeg.nii")

        print(phaseData.shape)
        reconInp = reconstructImageFromPatches(phaseData[:,0].cpu(), [patchIdxs], PATCH_SIZE)
        sitk.WriteImage(sitk.GetImageFromArray(reconInp.numpy()), "reconInp.nii")

    def inference(self, stateDictPath: str, outputPath: str = "./outputs", outputPathPCR: str = "./outputsPCR"):
        stateDictPath = os.path.join(self.outputFolder, stateDictPath)
        stateDict = torch.load(stateDictPath, map_location=self.model.device, weights_only=False)
        modelState = stateDict['networkWeights']
        epoch = stateDict["epoch"]
        print(f"Loading {os.path.basename(stateDictPath)} from epoch {epoch}")
        self.model.load_state_dict(modelState)
        self.model.eval()
        self.model.ret = "segOnly"

        outputPath = os.path.join(self.outputFolder, outputPath)
        outputPathPCR = os.path.join(self.outputFolder, outputPathPCR)
        os.makedirs(outputPath, exist_ok=True)
        os.makedirs(outputPathPCR, exist_ok=True)

        import pandas as pd
        from MAMAMIA.src.challenge.metrics import hausdorff_distance
        
        print("Running inference!")
        # TODO: Finish me
        scoreDF = pd.DataFrame(columns=["Patient ID", "Dice", "HD95", "PCR", "PCR Pred"])
        for struct in self.tsDataloader:
            obj, patientIDs = struct
            seg: dict = obj['seg']
            phase: dict = obj['phase']

            # print(seg)

            if type(patientIDs) == str:
                patientID = patientIDs
                patientIDs = [patientIDs]
            else:
                patientID = patientIDs[0]       # This is weird: TODO Fix ME
            

            target, patchIdxs = seg[0]()
            # pcrVal = pcr[i]()

            target: torch.Tensor = target.unsqueeze(0)          # add a singleton batch dimension

            # run the phase handles
            phaseData = [phase[p][0]() for p in phase.keys()]
            phaseData = torch.stack(phaseData).unsqueeze(0)     # add a singleton batch dimension

            segOut: torch.Tensor = self.model(phaseData, patientIDs, patchIdxs)
            # convert logits to binary prediction
            # this is equivalent to sigmoid(segOut) > 0.5
            segOut = (segOut > 0).int()

            segImageArr = reconstructImageFromPatches(segOut.transpose(1, 0), [patchIdxs], PATCH_SIZE)
            targetImageArr = reconstructImageFromPatches(target, [patchIdxs], PATCH_SIZE)
            
            dice = Dice(segImageArr, targetImageArr)

            segImageArr = segImageArr.detach().cpu().numpy()
            targetImageArr = targetImageArr.detach().cpu().numpy()

            hausdorff = hausdorff_distance(targetImageArr, segImageArr)
            
            row = pd.DataFrame({"Patient ID": [patientID], "Dice": [dice], "HD95": [hausdorff]})
            scoreDF = pd.concat([scoreDF, row])

            sitk.WriteImage(sitk.GetImageFromArray(segImageArr), 
                            os.path.join(outputPath, f"{patientID}_pred.nii"))
            sitk.WriteImage(sitk.GetImageFromArray(targetImageArr), 
                            os.path.join(outputPath, f"{patientID}.nii"))
            
            print(f"Finished patient {patientID}: Dice {dice:.4f}\tHausdorff 95 {hausdorff:.4f}", end="\r")
        print()
        scoreDF.to_csv(os.path.join(self.outputFolder, "scores.csv"))

        print(f"Average Dice: {scoreDF["Dice"].mean()}")
        print(f"Average HD95: {scoreDF["HD95"].mean()}")
        # from score_task1 import doScoring
        # doScoring(os.path.dirname(outputPath))
        
        
        # import pdb, pandas as pd, SimpleITK as sitk
        # pdb.set_trace()
        
        # from predictPCR import scorePCR
        # scorePCR(predPath)
        # from MAMAMIA.src.challenge.scoring_task2 import doScoring
        # doScoring(os.path.dirname(predPath))
    
    def plotLosses(self):
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot PCR Accuracy on the left y-axis
        ax1.plot(self.PCRPercentages, label='PCR Accuracy', color='blue')
        ax1.plot(self.PCRBalancedAccs, label='Balanced Accuracy', color='green')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('PCR Accuracy', fontsize=12, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue', labelsize=10)
        ax1.tick_params(axis='x', labelsize=10)

        # Create a second y-axis for BCE Loss on the right
        ax2 = ax1.twinx()
        ax2.plot(self.clsLosses, label='BCE Loss', color='red')
        ax2.set_ylabel('BCE Loss', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red', labelsize=10)

        # Add a title and adjust layout
        plt.title('PCR Prediction Accuracy and BCE Loss Over Epochs', fontsize=14)
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes, fontsize=10)
        fig.tight_layout()  # Adjust layout to prevent clipping

        # Save the plot
        plt.savefig(os.path.join(self.outputFolder, 'pcr_progress.png'))
        plt.clf()

    def saveModel(self, tag: str = None):
        stateDict = {"networkWeights": self.model.state_dict(),
                     "optimizerState": self.optimizer.state_dict(),
                     "gradScalerState": self.gradScaler.state_dict(),
                     "schedulerState": self.LRScheduler.state_dict(),
                     "epoch": self.currentEpoch}
        if tag is not None:
            torch.save(stateDict, f"{self.outputFolder}/{tag}.pth")
        torch.save(stateDict, f"{self.outputFolder}/Latest.pth")

if __name__ == "__main__":
    writer = SummaryWriter()
    datasetName = "Dataset106_cropped_Xch_breast_no_norm"
    dataDir = rf"E:\MAMA-MIA\my_preprocessed_data\{datasetName}"
    # pretrainedDecoderPath = "pretrained_weights/nnunet_pretrained_weights_64_best.pth"
    pretrainedDecoderPath = None
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    modelName = "conv_bottleneck_with_skips_full"
    trainer = MyTrainer(100, modelName, "Testing", joint=False)
    
    trainer.setup(dataDir, 
                  device, 
                  pretrainedDecoderPath=pretrainedDecoderPath, 
                  useSkips=True, 
                  bottleneck="Conv", 
                  test=False)
    trainer.train()
    trainer.inference("BestSeg.pth",
                      outputPath=rf"outputs/predSegmentationsCropped",
                      outputPathPCR=rf"outputs")
