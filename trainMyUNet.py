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
from helpers import *
from time import time

class MyTrainer():
    def __init__(self, nEpochs: int, modelName: str = "", tag: str = "", joint: bool = True, test: bool = False,
                 patientDataPath="clinical_and_imaging_info.xlsx"):
        self.nEpochs = nEpochs
        self.joint = joint
        self.pretrainSegmentation = self.nEpochs * 0.5
        self.peakLR = 1e-5
        self.minLR = 1e-10
        self.currentEpoch = 0
        self.oversampleFG = 0.3

        self.clsLosses = []
        self.PCRPercentages = []
        self.PCRBalancedAccs = []

        self.outputFolder = f"./transformerResults/{modelName}"
        os.makedirs(self.outputFolder, exist_ok=True)

        self.patientDataPath = os.path.join(os.environ.get("MAMAMIA_DATA"), patientDataPath)

        self.tag = tag
        self.test = test
        self.writer = None
        if not self.test:
            self.writer = SummaryWriter(os.path.join(self.outputFolder, f"log{tag}"))
        self.logGradients = True

    def setup(self,
              dataDir: str,
              device: torch.device,
              pretrainedDecoderPath: str = None,
              useSkips: bool = True,
              bottleneck: str = "MyTransformer"):

        self.device = device
        nHeads = 8
        patchSize = 32

        self.model = MyUNet(patchSize,
                            pretrainedDecoderPath=pretrainedDecoderPath,
                            patientDataPath=self.patientDataPath,
                            nHeads=nHeads,
                            useSkips=useSkips,
                            joint=self.joint,
                            bottleneck=bottleneck,
                            nBottleneckLayers=4).to(device)
        
        self.trDataloader, self.vlDataloader, self.tsDataloader = GetDataloaders(dataDir, device, self.patientDataPath, self.oversampleFG, test=self.test)

        # change optimizer and scheduler
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.encoder.parameters(), 'lr': self.minLR * 20, 'weight_decay': 1e-4},
            {'params': self.model.bottleneck.parameters(), 'lr': self.minLR, 'weight_decay': 1e-4},
            {'params': self.model.decoder.parameters(), 'lr': self.minLR * 10,  'weight_decay': 1e-4},
            {'params': self.model.classifier.parameters(), 'lr': self.minLR * 50,  'weight_decay': 1e-4}
        ])
        # self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.minLR, weight_decay=1e-4)
        # use much smaller initial scale since our early losses will be close to 1
        self.gradScaler = torch.GradScaler(device.type)
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

        # bce pos_weight should be close to the ratio of background to foreground in segmentations
        # since we oversample, we know some percentage of patches will have foreground
        # we don't know how much foreground though, since we sample randomly on the bounding box
        # so we guess that around 70% of each oversampled patch is foreground. -> self.oversampleFG * 0.7
        # If we dont oversample, then the average ratio of background to foreground is used
        # (i don't actually know this number rn, so again just guess that ~5% voxels are foreground)
        bcePosWeight = 1 / (self.oversampleFG * 0.7) if self.oversampleFG != 0 else 20
        self.SegLoss = SegLoss(bcePosWeight=bcePosWeight)

        return self

    def train(self, continueTraining: bool = False, modelName: str = None):
        if continueTraining:
            assert modelName is not None, "Cannot continue training if no model state is given."
            self.loadModel(modelName)
            print(f"Continuing training from epoch {self.currentEpoch}")
            self.nEpochs = self.currentEpoch + 1

        startEpoch = self.currentEpoch
        bestSeg = 10.
        bestPCR = 10.
        bestJoint = 10.

        import numpy.random as rd
        rd.seed(1234)
        print("Starting training!")
        start = time()
        for epoch in range(startEpoch, self.nEpochs):
            self.currentEpoch = epoch
            print(f"Epoch {self.currentEpoch}:")

            # =========================================
            #               TRAINING LOOP
            # =========================================

            diceLossesThisEpoch = []
            bceLossesThisEpoch = []
            bdLossesThisEpoch = []
            segLossesThisEpoch = []
            pcrLossesThisEpoch = []
            diceThisEpoch = []

            lr = self.optimizer.param_groups[0]["lr"]
            if self.writer:
                self.writer.add_scalar("LR", lr, self.currentEpoch)

            self.model.train()

            startEpoch = time()
            for idx, struct in enumerate(self.trDataloader):        # iterate over patient cases
                handle, patientIDs = struct

                if type(patientIDs) == str:
                    patientIDs = [patientIDs]
                # print(f"thing {idx}, {patientIDs}: {obj.keys()}")
                
                segs, dmaps, phases, pcrs, patchIndices = handle()

                torch.cuda.empty_cache()
                pcr: torch.Tensor       = pcrs[0]       # should be singleton tensor
                target: torch.Tensor    = segs[0].unsqueeze(1)
                distMap: torch.Tensor   = dmaps[0].unsqueeze(1)
                phases: torch.Tensor    = phases.unsqueeze(0)
                patchIndices = torch.tensor(patchIndices).to(self.device)

                with torch.autocast(self.device.type):
                    segOut, sharedFeatures, pcrOut = self.model(phases, patientIDs, patchIndices)
                    pcrLoss = None
                    if self.currentEpoch > self.pretrainSegmentation and pcrOut and self.joint:
                        pcrLoss: torch.Tensor = self.PCRloss(pcrOut, pcr)

                    bceLoss, diceLoss, bdLoss = self.SegLoss(segOut, target, distMap)
                    segLoss: torch.Tensor = bceLoss + diceLoss + bdLoss
                    
                    print(f"\t{segLoss:.4f} = BCE Loss: {bceLoss:.4f} + Dice Loss: {diceLoss:.4f} + BD Loss: {bdLoss:.4f}", end='\r')

                del segs, dmaps, phases, pcrs, patchIndices

                segLossesThisEpoch.append(segLoss.item())
                bceLossesThisEpoch.append(bceLoss.item())
                diceLossesThisEpoch.append(diceLoss.item())
                bdLossesThisEpoch.append(bdLoss.item())
                
                # now only do foreground for "real" Dice score
                segOut: torch.Tensor = (segOut > 0).int()
                dice = Dice(segOut.detach().cpu(), target.detach().cpu())
                diceThisEpoch.append(dice)

                # free as much space as possible before backward()
                del segOut, pcrOut, target, distMap, bceLoss, diceLoss, bdLoss
                
                self.optimizer.zero_grad()
                if self.joint and pcrLoss:
                    assert sharedFeatures is not None, "Cannot do joint backward without shared features!"
                    pcrLossesThisEpoch.append(pcrLoss.item())

                    losses = self.gradScaler.scale([segLoss, pcrLoss])
                    # losses = [segLoss, pcrLoss]
                    mtl_backward(losses=losses, 
                                 features=sharedFeatures, 
                                 aggregator=self.aggregator,
                                 tasks_params=[list(self.model.decoder.parameters()), 
                                               list(self.model.classifier.parameters())],
                                 shared_params=list(self.model.encoder.parameters()) + list(self.model.bottleneck.parameters()))
                else:
                    scaledLoss = self.gradScaler.scale(segLoss)
                    scaledLoss.backward()
                    # segLoss.backward()
                    
                self.gradScaler.step(self.optimizer)
                self.gradScaler.update()

                del segLoss, pcrLoss, sharedFeatures

            print()
            if self.writer:
                self.writer.add_scalar('Dice Loss/Training', Mean(diceLossesThisEpoch), self.currentEpoch)
                self.writer.add_scalar('BCE Loss/Training', Mean(bceLossesThisEpoch), self.currentEpoch)
                self.writer.add_scalar('Boundary Loss/Training', Mean(bdLossesThisEpoch), self.currentEpoch)
                self.writer.add_scalar('Overall Seg Loss/Training', Mean(segLossesThisEpoch), self.currentEpoch)
                self.writer.add_scalar('Dice/Training', Mean(diceThisEpoch), self.currentEpoch)
                
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
            diceLossesVal = []
            bceLossesVal = []
            bdLossesVal = []
            pcrLossesVal = []
            diceVal = []
            self.model.eval()

            for idx, struct in enumerate(self.vlDataloader):        # iterate over patient cases
                handle, patientIDs = struct

                if type(patientIDs) == str:
                    patientIDs = [patientIDs]
                # print(f"thing {idx}, {patientIDs}: {obj.keys()}")
                
                segs, dmaps, phases, pcrs, patchIndices = handle()

                torch.cuda.empty_cache()
                pcr: torch.Tensor       = pcrs[0]       # should be singleton tensor
                target: torch.Tensor    = segs[0].unsqueeze(1)
                distMap: torch.Tensor   = dmaps[0].unsqueeze(1)
                phases: torch.Tensor    = phases.unsqueeze(0)
                patchIndices = torch.tensor(patchIndices).to(self.device)

                with torch.autocast(self.device.type):
                    segOut, _, pcrOut = self.model(phases, patientIDs, patchIndices)
                    pcrLoss = None
                    if self.currentEpoch > self.pretrainSegmentation and pcrOut and self.joint:
                        pcrLoss: torch.Tensor = self.PCRloss(pcrOut, pcr)

                    # do one-hot encoding since that makes our loss functions behave better
                    bceLoss, diceLoss, bdLoss = self.SegLoss(segOut, target, distMap)
                    segLoss: torch.Tensor = bceLoss + diceLoss + bdLoss

                del segs, dmaps, phases, pcrs, patchIndices
                
                segLossesVal.append(segLoss.item())
                bceLossesVal.append(bceLoss.item())
                diceLossesVal.append(diceLoss.item())
                bdLossesVal.append(bdLoss.item())
                
                # now only do foreground for "real" Dice score
                segOut: torch.Tensor = (segOut > 0).int()
                dice = Dice(segOut.detach().cpu(), target.detach().cpu())
                diceVal.append(dice)

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

                del segOut, pcrOut, target, bceLoss, diceLoss, bdLoss, segLoss, pcrLoss
            
            self.LRScheduler.step()

            # MODEL CHECKPOINTING
            if self.writer:
                avgSegValLoss = Mean(segLossesVal)
                
                self.writer.add_scalar('Overall Seg Loss/Validation', avgSegValLoss, self.currentEpoch)

                self.writer.add_scalar('Dice Loss/Validation', Mean(diceLossesVal), self.currentEpoch)
                self.writer.add_scalar('BCE Loss/Validation', Mean(bceLossesVal), self.currentEpoch)
                self.writer.add_scalar('Boundary Loss/Validation', Mean(bdLossesVal), self.currentEpoch)

                avgDiceVal = Mean(diceVal)
                self.writer.add_scalar('Dice/Validation', avgDiceVal, self.currentEpoch)

                if avgSegValLoss < bestSeg:
                    bestSeg = avgSegValLoss
                    print(f"Saving Best Seg: epoch {self.currentEpoch}")
                    self.saveModel(f"BestSeg{self.tag}")
                
                if len(pcrLossesVal) > 0:
                    avgPCRValLoss = Mean(pcrLossesVal)
                    self.writer.add_scalar('PCR Loss/Validation', avgPCRValLoss, self.currentEpoch)

                    avgJoint = Mean([s + p for s, p in zip(segLossesVal, pcrLossesVal)])
                    self.writer.add_scalar('Joint Loss/Validation', avgJoint, self.currentEpoch)

                    if avgPCRValLoss < bestPCR:
                        bestPCR = avgPCRValLoss
                        print(f"Saving Best PCR: epoch {self.currentEpoch}")
                        self.saveModel(f"BestPCR{self.tag}")
                    
                    if avgJoint < bestJoint:
                        bestJoint = avgJoint
                        print(f"Saving Best Joint: epoch {self.currentEpoch}")
                        self.saveModel(f"BestJoint{self.tag}")

            if self.currentEpoch % 5 == 0:
                self.saveModel()

            print(f"\tFull loop took {FormatSeconds(time() - startEpoch)}")
        # print(patchIdxs)

        print("Done training!")
        print(f"\tTook {FormatSeconds(time() - start)}.")
        return

    def inference(self, stateDictPath: str, outputPath: str = "predSegmentationsCropped", outputPathPCR: str = "predPCR"):
        stateDictPath = os.path.join(self.outputFolder, stateDictPath)
        stateDict = torch.load(stateDictPath, map_location=self.device, weights_only=False)
        modelState = stateDict['networkWeights']
        epoch = stateDict["epoch"]
        print(f"Loading {os.path.basename(stateDictPath)} from epoch {epoch}")
        self.model.load_state_dict(modelState)
        self.model.eval()
        self.model.ret = "segOnly"

        outputPath = os.path.join(self.outputFolder, f"outputs{self.tag}", outputPath)
        outputPathPCR = os.path.join(self.outputFolder, f"outputs{self.tag}", outputPathPCR)
        os.makedirs(outputPath, exist_ok=True)
        os.makedirs(outputPathPCR, exist_ok=True)

        import pandas as pd
        from MAMAMIA.src.challenge.metrics import hausdorff_distance
        
        print("Running inference!")
        # TODO: Finish me
        scoreDF = pd.DataFrame(columns=["Patient ID", "Dice", "HD95", "PCR", "PCR Pred"])
        for struct in self.tsDataloader:
            handle, patientID = struct

            # print(f"thing {idx}, {patientIDs}: {obj.keys()}")
            
            segs, _, phases, pcrs, patchIndices = handle()

            torch.cuda.empty_cache()
            # pcr: torch.Tensor       = pcrs[0].detach().cpu()       # should be singleton tensor
            target: torch.Tensor    = segs[0].int().detach().cpu()
            phase1: torch.Tensor    = phases[1].float().detach().cpu()
            phases: torch.Tensor    = phases.unsqueeze(0)
            patchIndices = torch.tensor(patchIndices).to(self.device)

            with torch.autocast(self.device.type):
                n = len(patchIndices)
                allOuts = []
                for startI in range(0, n, CHUNK_SIZE):
                    stopI = min(startI + CHUNK_SIZE, n)
                    segOut: torch.Tensor = self.model(phases[:, :, startI:stopI], [patientID], patchIndices[startI:stopI])
                    allOuts.append((segOut > 0).int().detach().cpu())
                    del segOut
            
            segOut = torch.cat(allOuts).squeeze()

            dicePatches = Dice(segOut, target)

            patchIndices = patchIndices.detach().cpu()
            phaseImageArr = reconstructImageFromPatches([phase1], [patchIndices], PATCH_SIZE)
            segImageArr = reconstructImageFromPatches([segOut], [patchIndices], PATCH_SIZE)
            targetImageArr = reconstructImageFromPatches([target], [patchIndices], PATCH_SIZE)
            
            dice = Dice(segImageArr, targetImageArr)

            segImageArr = segImageArr.numpy()
            targetImageArr = targetImageArr.numpy()

            hausdorff = hausdorff_distance(targetImageArr, segImageArr)
            
            row = pd.DataFrame({"Patient ID": [patientID], "Dice (Full Image)": [dice], "Dice (Avg Over Patches)": [dicePatches],"HD95": [hausdorff]})
            scoreDF = pd.concat([scoreDF, row])

            sitk.WriteImage(sitk.GetImageFromArray(segImageArr), 
                            os.path.join(outputPath, f"{patientID}_pred.nii"))
            sitk.WriteImage(sitk.GetImageFromArray(targetImageArr), 
                            os.path.join(outputPath, f"{patientID}.nii"))
            sitk.WriteImage(sitk.GetImageFromArray(phaseImageArr),
                            os.path.join(outputPath, f"{patientID}_phase.nii"))
            
            print(f"Finished patient {patientID}:\tDice {dice:.4f}\tHausdorff 95 {hausdorff:.4f}")
        print()
        scoreDF.to_csv(os.path.join(self.outputFolder, f"outputs{self.tag}", "scores.csv"))

        print(f"Average Dice: {scoreDF["Dice (Full Image)"].mean()}")
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

    def saveModel(self, saveAs: str = None):
        stateDict = {"networkWeights": self.model.state_dict(),
                     "optimizerState": self.optimizer.state_dict(),
                     "gradScalerState": self.gradScaler.state_dict(),
                     "schedulerState": self.LRScheduler.state_dict(),
                     "epoch": self.currentEpoch}
        if saveAs is not None:
            torch.save(stateDict, f"{self.outputFolder}/{saveAs}.pth")
        torch.save(stateDict, f"{self.outputFolder}/Latest{self.tag}.pth")
    
    def loadModel(self, modelName: str):
        stateDict = torch.load(os.path.join(self.outputFolder, modelName))
        self.model.load_state_dict(stateDict["networkWeights"])
        self.optimizer.load_state_dict(stateDict["optimizerState"])
        self.gradScaler.load_state_dict(stateDict["gradScalerState"])
        self.LRScheduler.load_state_dict(stateDict["schedulerState"])
        self.currentEpoch = stateDict["epoch"]

if __name__ == "__main__":
    writer = SummaryWriter()
    datasetName = "Dataset106_cropped_Xch_breast_no_norm"
    # dataDir = rf"F:\MAMA-MIA\my_preprocessed_data\{datasetName}"
    dataDir = rf"/mnt/storageSSD/MAMA-MIA/data/my_preprocessed_data/{datasetName}"
    # pretrainedDecoderPath = "pretrained_weights/nnunet_pretrained_weights_64_best.pth"
    pretrainedDecoderPath = None
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    bottleneck = BOTTLENECK_TRANSFORMERST
    # bottleneck = BOTTLENECK_CONV
    skips = True
    joint = False
    test  = False        # testing the model on a few specific patients so we don't have to wait for the dataloader
    modelName = f"{bottleneck}{"Joint" if joint else ""}{"With" if skips else "No"}Skips{"-TEST" if test else ""}"
    trainer = MyTrainer(nEpochs=100, modelName=modelName, tag="FixedLosses", joint=joint, test=test)
    
    trainer.setup(dataDir, 
                  device, 
                  pretrainedDecoderPath=pretrainedDecoderPath, 
                  useSkips=True, 
                  bottleneck=bottleneck)       
    # trainer.train(continueTraining=True, modelName="LatestFullChannels.pth")
    trainer.train()
    trainer.inference(f"Latest{trainer.tag}.pth")
    # trainer.inference(f"BestSeg{trainer.tag}.pth")
