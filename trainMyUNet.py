import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from time import time
from torch.utils.tensorboard import SummaryWriter
from torchjd.aggregation import UPGrad
from torchjd import mtl_backward
from sklearn.metrics import roc_auc_score
from MyUNet import MyUNet
from Losses import PCRLoss, PCRLossWithConfidence, SegLoss, Dice, tp_fp_tn_fn, GetMetrics
from Schedulers import WarmupCosineAnnealingWithRestarts
from CustomLoader import GetDataloaders
from DataProcessing import ReconstructImageFromPatches, GetPatches
from helpers import *
from Augmenter import GetNoTransforms, GetTrainingTransforms

class MyTrainer():
    def __init__(self, nEpochs: int, modelName: str = "", tag: str = "", joint: bool = True, stopEarly: int = None, 
                 useJD: bool = False, patientDataPath="clinical_and_imaging_info.xlsx", test: bool = False):
        self.nEpochs = nEpochs
        self.stopAfterPlateauEpochs = stopEarly if stopEarly else 10_000_000
        self.joint = joint
        self.useJD = useJD
        
        # Parameters to change!
        self.pretrainSegmentation = self.nEpochs * 0.5
        # self.pretrainSegmentation = 0
        self.pcrConfidence = False
        self.peakLR = 1e-5
        self.minLR = 1e-8
        self.currentEpoch = 0
        self.oversampleFG = 0.2
        self.oversampleRadius = 0.2
        self.batchSize = 2
        self.clipGrad = False
        self.downsample = 2

        self.trainingCompose = GetTrainingTransforms()
        self.valTestCompose = GetNoTransforms()

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
        nHeads = 16
        nBottleneckLayers = 12

        self.model = MyUNet(expectedPatchSize=PATCH_SIZE,
                            expectedChannels=[1, 64, 128, 256, 384, 576],
                            expectedStride=[2, 2, 2, 2, 2],
                            pretrainedDecoderPath=pretrainedDecoderPath,
                            patientDataPath=self.patientDataPath,
                            nHeads=nHeads,
                            useSkips=useSkips,
                            joint=self.joint,
                            pcrConfidence=self.pcrConfidence,
                            bottleneck=bottleneck,
                            nBottleneckLayers=nBottleneckLayers).to(device)

        dataTime = time()
        self.trDataloader, self.vlDataloader, self.tsDataloader = GetDataloaders(dataDir, self.patientDataPath, self.trainingCompose, self.valTestCompose,
                                                                                 self.downsample, batchSize=self.batchSize, shuffle=True, test=self.test)
        print(f"\tTook {FormatSeconds(time() - dataTime)}")

        # change optimizer and scheduler
        decoderLRWeight = 50
        if pretrainedDecoderPath:
            decoderLRWeight = 0.1
        
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.encoder.parameters(), 'lr': self.minLR * 50, 'weight_decay': 1e-4},
            {'params': self.model.bottleneck.parameters(), 'lr': self.minLR, 'weight_decay': 1e-5},
            {'params': self.model.decoder.parameters(), 'lr': self.minLR * decoderLRWeight,  'weight_decay': 1e-4},
            {'params': self.model.classifier.parameters(), 'lr': self.minLR * 10,  'weight_decay': 1e-3}
        ])
        # self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.minLR, weight_decay=1e-4)

        self.gradScaler = torch.GradScaler(device.type)
        self.aggregator = UPGrad()

        warmupPercent = 0.05
        nCycles = 2
        nWarmupSteps = round(warmupPercent * self.nEpochs)
        nCycleSteps = round((1 - warmupPercent) * self.nEpochs / nCycles) + 1

        self.LRScheduler = WarmupCosineAnnealingWithRestarts(
            self.optimizer,
            warmup_steps=nWarmupSteps,
            cycle_steps=nCycleSteps,
            maxLR=self.peakLR,
            minLR=self.minLR,
            damping=0.5
        )

        self.PCRloss = PCRLoss() if not self.pcrConfidence else PCRLossWithConfidence()

        # bce pos_weight should be close to the ratio of background to foreground in segmentations
        # since we oversample, we know some percentage of patches will have foreground
        # we don't know how much foreground though, since we sample randomly on the bounding box
        # so we guess that around 40% of each oversampled patch is foreground. -> self.oversampleFG * 0.4
        #       (From looking at samples in the debugger, 20% is not a bad guess)
        # If we dont oversample, then the average ratio of background to foreground is used
        # (i don't actually know this number rn, so again just guess that ~10% voxels are foreground)
        # So otherwise, posWeight = 90% / 10% = 9
        bcePosWeight = min(1 / (self.oversampleFG * 0.4) - 1, 15) if self.oversampleFG != 0 else 9
        self.SegLoss = SegLoss(bcePosWeight=bcePosWeight, downsample=self.downsample)

    def train(self, continueTraining: bool = False, modelName: str = None):
        if continueTraining:
            assert modelName is not None, "Cannot continue training if no model state is given."
            self.loadModel(modelName)
            print(f"Continuing training from epoch {self.currentEpoch}")
            self.nEpochs = max(self.currentEpoch + 1, self.nEpochs)
        
        startEpoch = self.currentEpoch
        bestSeg = 0.
        bestPCR = 1000.
        bestJoint = 1000.

        bestSegEpoch = self.currentEpoch
        bestJointEpoch = self.currentEpoch

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
            nBatches = len(self.trDataloader)

            # iterations = ["oversample", "oversample", "oversample", "no tumor", "no tumor", "no tumor"]
            iterations = ["oversample", "oversample", "oversample", "no tumor", "no tumor"]
            nHandles = len(iterations)
            for idx, struct in enumerate(self.trDataloader):        # iterate over patient cases
                mris, dmap, seg, pcr, bbox, patientIDs = zip(*struct)
                rd.shuffle(iterations)
                for i, it in enumerate(iterations):
                    if it == "oversample":
                        args = [mris, dmap, seg, PATCH_SIZE, NUM_PATCHES, self.oversampleFG,
                                self.oversampleRadius, bbox, self.downsample, False]
                    else:
                        args = [mris, dmap, seg, PATCH_SIZE, NUM_PATCHES, -1, 0, bbox, self.downsample, False]
                    phases, distMap, target, patchIndices = GetPatches(*args)
                    torch.cuda.empty_cache()
                    phases: torch.Tensor    = phases.transpose(1, 2).to(self.device, non_blocking=True)
                    distMap: torch.Tensor   = distMap.to(self.device, non_blocking=True)
                    target: torch.Tensor    = target.to(self.device, non_blocking=True)
                    patchIndices            = patchIndices.to(self.device)

                    with torch.autocast(self.device.type):
                        segOut, sharedFeatures, pcrOut = self.model(phases, patientIDs, patchIndices)
                        pcrLoss = None
                        if self.currentEpoch >= self.pretrainSegmentation and pcrOut is not None and self.joint:
                            pcrLoss: torch.Tensor = self.PCRloss(pcrOut, pcr)

                        bceLoss, diceLoss, bdLoss = self.SegLoss(segOut, target, distMap)
                        segLoss: torch.Tensor = bceLoss + bdLoss

                        # Dice loss bad on foreground channels when we only background
                        # So ignore it for these batches
                        if target.sum().item() > 0:
                            segLoss += diceLoss
                            diceLossesThisEpoch.append(diceLoss.item())

                        print(f"\tTraining Batch {idx + (1 + i) / nHandles:.2f}/{nBatches:.2f}: {segLoss:.4f} = BCE Loss: {bceLoss:.4f} + Dice Loss: {diceLoss:.4f} + BD Loss: {bdLoss:.4f}{f" + PCR Loss {pcrLoss:.4f}" if pcrLoss is not None else ""}", end='\r')

                    del phases, distMap, patchIndices

                    segLossesThisEpoch.append(segLoss.item())
                    bceLossesThisEpoch.append(bceLoss.item())
                    bdLossesThisEpoch.append(bdLoss.item())
                    
                    # now only do foreground for "real" Dice score
                    segOut: torch.Tensor = (segOut > 0).int()
                    dice = Dice(segOut.detach().cpu(), target.detach().cpu())
                    diceThisEpoch.append(dice)

                    # free as much space as possible before backward()
                    del segOut, pcrOut, target, bceLoss, diceLoss, bdLoss
                    
                    self.optimizer.zero_grad()
                    if self.joint and pcrLoss and not pcrLoss.isnan().any():
                        if self.useJD:
                            assert sharedFeatures is not None, "Cannot do joint backward without shared features!"
                            pcrLossesThisEpoch.append(pcrLoss.item())

                            losses = self.gradScaler.scale([segLoss, pcrLoss])
                            mtl_backward(losses=losses, 
                                        features=sharedFeatures, 
                                        aggregator=self.aggregator,
                                        tasks_params=[list(self.model.decoder.parameters()), 
                                                    list(self.model.classifier.parameters())],
                                        shared_params=list(self.model.encoder.parameters()) + list(self.model.bottleneck.parameters()))
                        else:
                            scaledLoss = self.gradScaler.scale(segLoss + pcrLoss)
                            scaledLoss.backward()
                    else:
                        scaledLoss = self.gradScaler.scale(segLoss)
                        scaledLoss.backward()
                    
                    if self.clipGrad:
                        self.gradScaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=12.)
                    
                    self.gradScaler.step(self.optimizer)
                    self.gradScaler.update()

                    del segLoss, pcrLoss, sharedFeatures
                del mris, dmap, seg, pcr

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
            diceValFull = []
            diceValPatches = []
            sensVal = []
            specVal = []
            truePCRs = []
            predPCRs = []
            self.model.eval()

            nBatches = len(self.vlDataloader)
            for idx, struct in enumerate(self.vlDataloader):        # iterate over patient cases
                mris, dmap, seg, pcr, bbox, patientIDs = zip(*struct)
                truePCRs += [t.item() for t in pcr]
                phases, distMap, target, patchIndices = GetPatches(mris, dmap, seg, PATCH_SIZE, NUM_PATCHES, 0, 0, bbox, self.downsample, True)
                torch.cuda.empty_cache()
                phases: torch.Tensor    = phases.transpose(1, 2).to(self.device, non_blocking=True)
                distMap: torch.Tensor   = distMap
                target: torch.Tensor    = target.int()
                patchIndices            = patchIndices.to(self.device)

                with torch.autocast(self.device.type):
                    n = patchIndices.shape[1]
                    allOuts = []
                    allPCRs = []
                    for startI in range(0, n, CHUNK_SIZE):
                        stopI = min(startI + CHUNK_SIZE, n)
                        segOut, _, pcrOut = self.model(phases[:, :, startI:stopI], patientIDs, patchIndices[:, startI:stopI])
                        pcrLoss = None

                        allOuts.append(segOut.detach().cpu())
                        if self.currentEpoch >= self.pretrainSegmentation and pcrOut is not None and self.joint:
                            pcrLoss: torch.Tensor = self.PCRloss(pcrOut, pcr)
                            if self.pcrConfidence:
                                allPCRs.append((torch.sigmoid(pcrOut[0]) * torch.sigmoid(pcrOut[1])).squeeze(dim=-1).detach().cpu())
                            else:
                                allPCRs.append(pcrOut.squeeze(dim=-1).detach().cpu())
                            

                        del segOut
                
                    segOut = torch.cat(allOuts, dim=1)
                    bceLoss, diceLoss, bdLoss = self.SegLoss(segOut, target, distMap)
                    segLoss: torch.Tensor = bceLoss + diceLoss + bdLoss

                    if self.joint:
                        predPCRs.append(Mean(allPCRs))

                patchIndices = patchIndices.detach().cpu()

                print(f"\tValidation Batch {idx+1}/{nBatches}: {segLoss:.4f} = BCE Loss: {bceLoss:.4f} + Dice Loss: {diceLoss:.4f} + BD Loss: {bdLoss:.4f}{f" + PCR Loss {pcrLoss:.4f}" if pcrLoss is not None else ""}", end='\r')

                del phases, distMap
                
                segLossesVal.append(segLoss.item())
                bceLossesVal.append(bceLoss.item())
                diceLossesVal.append(diceLoss.item())
                bdLossesVal.append(bdLoss.item())
                
                segOut: torch.Tensor = (segOut > 0).int().detach().cpu()
                target = target.detach().cpu()
                dicePatches = []
                diceFull = []
                sens = []
                spec = []
                for i in range(len(patientIDs)):
                    dicePatches.append(Dice(segOut[i], target[i]))

                    segImageArr     = ReconstructImageFromPatches(segOut[i], patchIndices[i], PATCH_SIZE)
                    targetImageArr  = ReconstructImageFromPatches(target[i], patchIndices[i], PATCH_SIZE)
                    
                    diceFull.append(Dice(segImageArr, targetImageArr))
                    tp, fp, tn, fn = tp_fp_tn_fn(segImageArr, targetImageArr)
                    sens.append(tp / (tp + fn))
                    spec.append(tn / (tn + fp))

                diceValFull.append(Mean(diceFull))
                diceValPatches.append(Mean(dicePatches))
                sensVal.append(Mean(sens))
                specVal.append(Mean(spec))

                if self.joint and pcrLoss is not None and not pcrLoss.isnan().any():
                    pcrLossesVal.append(pcrLoss.item())

                del segOut, pcrOut, target, patchIndices, bceLoss, diceLoss, bdLoss, segLoss, pcrLoss, mris, dmap, seg, pcr
            
            self.LRScheduler.step()
            print()

            auc = None
            if self.joint and self.currentEpoch >= self.pretrainSegmentation:
                predPCRs = torch.cat(predPCRs).tolist()
                # Convert to numpy arrays (if they aren’t already)
                truePCRs = np.array(truePCRs)
                predPCRs = np.array(predPCRs)

                # Mask out invalid entries
                mask = truePCRs != -1
                truePCRs_masked = truePCRs[mask]
                predPCRs_masked = predPCRs[mask]

                # Now compute AUC only on valid entries
                auc = roc_auc_score(truePCRs_masked, predPCRs_masked)

            # MODEL CHECKPOINTING
            if self.writer:
                avgSegValLoss = Mean(segLossesVal)
                
                self.writer.add_scalar('Overall Seg Loss/Validation', avgSegValLoss, self.currentEpoch)

                self.writer.add_scalar('Dice Loss/Validation', Mean(diceLossesVal), self.currentEpoch)
                self.writer.add_scalar('BCE Loss/Validation', Mean(bceLossesVal), self.currentEpoch)
                self.writer.add_scalar('Boundary Loss/Validation', Mean(bdLossesVal), self.currentEpoch)

                avgDiceVal = Mean(diceValFull)
                self.writer.add_scalar('Dice/Validation - Full', avgDiceVal, self.currentEpoch)
                self.writer.add_scalar('Dice/Validation - Patches', Mean(diceValPatches), self.currentEpoch)

                self.writer.add_scalar('Sensitivity - Val', Mean(sensVal), self.currentEpoch)
                self.writer.add_scalar('Specificity - Val', Mean(specVal), self.currentEpoch)

                if auc:
                    self.writer.add_scalar("PCR AUC", auc, self.currentEpoch)

                if avgDiceVal > bestSeg:
                    bestSeg = avgDiceVal
                    bestSegEpoch = self.currentEpoch
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
                        bestJointEpoch = self.currentEpoch
                        print(f"Saving Best Joint: epoch {self.currentEpoch}")
                        self.saveModel(f"BestJoint{self.tag}")

            if self.currentEpoch % 5 == 0 or (self.joint and self.currentEpoch == self.pretrainSegmentation - 1):
                self.saveModel()

            if self.joint and self.currentEpoch >= self.pretrainSegmentation and self.currentEpoch - bestJointEpoch >= self.stopAfterPlateauEpochs:
                print(f"Haven't seen joint task improvement in {self.stopAfterPlateauEpochs} epochs. Stopping early.")
                break
            elif not self.joint and self.currentEpoch - bestSegEpoch >= self.stopAfterPlateauEpochs:
                print(f"Haven't seen segmentation improvement in {self.stopAfterPlateauEpochs} epochs. Stopping early.")
                break

            print(f"\tFull loop took {FormatSeconds(time() - startEpoch)}")
        # print(patchIdxs)

        self.saveModel()
        print("Done training!")
        print(f"\tTook {FormatSeconds(time() - start)}.")
        return

    def inference(self, stateDictPath: str, resultsTag: str,
                  outputPath: str = "predSegmentationsCropped", outputPathPCR: str = "predPCR"):
        stateDictPath = os.path.join(self.outputFolder, stateDictPath)
        try:
            stateDict = torch.load(stateDictPath, map_location=self.device, weights_only=False)
        except:
            print(f"Failed to load {stateDictPath}, loading Latest model instead.")
            try:
                stateDictPath = os.path.join(self.outputFolder, f"Latest{self.tag}.pth")
                resultsTag = "Latest"
                stateDict = torch.load(stateDictPath, map_location=self.device, weights_only=False)
            except:
                print(f"Failed to load latest model! Did you run a model with the tag '{self.tag}'?")
                return
        modelState = stateDict['networkWeights']
        epoch = stateDict["epoch"]
        print(f"Loading {os.path.basename(stateDictPath)} from epoch {epoch}")
        self.model.load_state_dict(modelState)
        self.model.eval()
        if not self.joint:
            self.model.ret = "segOnly"

        resultsFolder = f"outputs{self.tag}{resultsTag}"
        outputPath = os.path.join(self.outputFolder, resultsFolder, outputPath)
        outputPathPCR = os.path.join(self.outputFolder, resultsFolder, outputPathPCR)
        os.makedirs(outputPath, exist_ok=True)
        os.makedirs(outputPathPCR, exist_ok=True)

        import pandas as pd
        from MAMAMIA.src.challenge.metrics import hausdorff_distance
        
        print("Running inference!")
        scoreDF = None
        for struct in self.tsDataloader:
            phases, dmap, seg, pcr, bbox, patientIDs = zip(*struct)
            phases, _, target, patchIndices = GetPatches(phases, dmap, seg, PATCH_SIZE, NUM_PATCHES, 0, 0, bbox, self.downsample, True)

            torch.cuda.empty_cache()
            target: torch.Tensor    = target.int()
            phases: torch.Tensor    = phases.transpose(1, 2).to(self.device, dtype=DTYPE_PHASE, non_blocking=True)
            phase1: torch.Tensor    = phases[:, 1].float().detach().cpu()
            patchIndices            = patchIndices.to(self.device)

            with torch.autocast(self.device.type):
                n = patchIndices.shape[1]
                allOuts = []
                allPCRs = []
                for startI in range(0, n, CHUNK_SIZE):
                    stopI = min(startI + CHUNK_SIZE, n)
                    x: torch.Tensor = self.model(phases[:, :, startI:stopI], patientIDs, patchIndices[:, startI:stopI])
                    if self.joint:
                        segOut, _, pcrOut = x
                        if self.pcrConfidence:
                            allPCRs.append((torch.sigmoid(pcrOut[0]) * torch.sigmoid(pcrOut[1])).detach().cpu())
                        else:
                            allPCRs.append(pcrOut.detach().cpu())
                        del pcrOut
                    else:
                        segOut = x
                    allOuts.append((segOut > 0).int().detach().cpu())
                    del segOut
            
            segOut = torch.cat(allOuts, dim=1)
            if self.joint:
                pcrOut = torch.cat(allPCRs, dim=-1)
                pcrOut = pcrOut.mean(dim=-1)

            patchIndices = patchIndices.detach().cpu()
            for i, patientID in enumerate(patientIDs):
                dicePatches = Dice(segOut[i], target[i])
                phaseImageArr   = ReconstructImageFromPatches(phase1[i], patchIndices[i], PATCH_SIZE)
                segImageArr     = ReconstructImageFromPatches(segOut[i], patchIndices[i], PATCH_SIZE)
                targetImageArr  = ReconstructImageFromPatches(target[i], patchIndices[i], PATCH_SIZE)
                
                dice = Dice(segImageArr, targetImageArr)
                segMetrics = GetMetrics(*tp_fp_tn_fn(segImageArr, targetImageArr), "Seg")

                segImageArr = segImageArr.numpy()
                targetImageArr = targetImageArr.numpy()

                hausdorff = min(hausdorff_distance(targetImageArr, segImageArr), 1000)

                row = {"Patient ID": [patientID], "Dice (Full Image)": [dice], "Dice (Avg Over Patches)": [dicePatches],"HD95": [hausdorff]} | segMetrics
                if self.joint:
                    row = row | {"PCR Pred": pcrOut[i].item(), "PCR True": pcr[i].item()}
                
                row = pd.DataFrame(row)
                if scoreDF is not None:
                    scoreDF = pd.concat([scoreDF, row])
                else:
                    scoreDF = row

                sitk.WriteImage(sitk.GetImageFromArray(segImageArr), 
                                os.path.join(outputPath, f"{patientID}_pred.nii"))
                sitk.WriteImage(sitk.GetImageFromArray(targetImageArr), 
                                os.path.join(outputPath, f"{patientID}.nii"))
                sitk.WriteImage(sitk.GetImageFromArray(phaseImageArr.numpy()),
                                os.path.join(outputPath, f"{patientID}_phase.nii"))
                
                print(" " * 80, end="\r")
                print(f"Finished patient {patientID}\tDice: {dice:.4f}\tHausdorff 95: {hausdorff:.4f}", end="\r")
        print()
        if self.joint:
                        # Convert to numpy arrays (if they aren’t already)
            truePCRs = np.array(scoreDF["PCR True"])
            predPCRs = np.array(scoreDF["PCR Pred"])

            # Mask out invalid entries
            mask = truePCRs != -1
            truePCRs_masked = truePCRs[mask]
            predPCRs_masked = predPCRs[mask]
            auc = roc_auc_score(truePCRs_masked, predPCRs_masked)
            scoreDF["AUC"] = auc

        savePath = os.path.join(self.outputFolder, resultsFolder, "scores.csv")
        scoreDF.to_csv(savePath, index=False)

        print(f"Saved results to: {savePath}")
        metrics = [col for col in scoreDF.columns if col != "Patient ID"]
        for metric in metrics:
            meanVal = scoreDF[metric].mean()
            stdVal = scoreDF[metric].std()
            print(f"Average {metric}:\t{meanVal:.4f} +/- {stdVal:.4f}")

# TODO: Finish me
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
        sd: dict[str, torch.Tensor] = stateDict["networkWeights"]
        ksToDelete = [k for k in sd.keys() if "bottleneck.patientDataEmbed" in k]
        for k in ksToDelete:
            v = sd[k]
            ks = k.split('.')
            ks.insert(1, "patientDataMod")
            sd['.'.join(ks)] = v
            del sd[k]
        self.model.load_state_dict(sd)
        self.optimizer.load_state_dict(stateDict["optimizerState"])
        self.gradScaler.load_state_dict(stateDict["gradScalerState"])
        self.LRScheduler.load_state_dict(stateDict["schedulerState"])
        self.currentEpoch = stateDict["epoch"] + 1

if __name__ == "__main__":
    writer = SummaryWriter()
    datasetName = "Dataset106_cropped_Xch_breast_no_norm"
    # dataDir = rf"F:\MAMA-MIA\my_preprocessed_data\{datasetName}"
    dataDir = rf"{os.environ.get("MAMAMIA_DATA")}/my_preprocessed_data/{datasetName}"
    pretrainedDecoderPath = r"transformerResults\TransformerTSJointWithSkips\BestSegOct20-DownsampleImages.pth"
    # pretrainedDecoderPath = None
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tag = "Oct29-DownsamplePoolPCRChunks"
    # tag = "Oct24-DownsampleImagesWithPCR"
    bottleneck = BOTTLENECK_SPATIOTEMPORAL
    # bottleneck = BOTTLENECK_TRANSFORMERTS
    # bottleneck = BOTTLENECK_TRANSFORMERST
    # bottleneck = BOTTLENECK_CONV
    skips = True
    joint = True
    test  = False        # testing the model on a few specific patients so we don't have to wait for the dataloader
    modelName = f"{bottleneck}{"Joint" if joint else ""}{"With" if skips else "No"}Skips" #{"-TEST" if test else ""}"
    trainer = MyTrainer(nEpochs=200, modelName=modelName, tag=tag, joint=joint, useJD=True, test=test)
    
    trainer.setup(dataDir, 
                  device, 
                  pretrainedDecoderPath=pretrainedDecoderPath, 
                  useSkips=skips, 
                  bottleneck=bottleneck)
    print(f"Set up model {modelName}")

    # trainer.train(continueTraining=True, modelName=f"Latest{tag}.pth")
    trainer.train()
    # trainer.inference(f"Latest{tag}.pth", "Latest")
    if joint:
        trainer.inference(f"BestPCR{tag}.pth", "BestPCR")
    else:
        trainer.inference(f"BestSeg{tag}.pth", "BestSeg")
