import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from time import time
from torch.utils.tensorboard import SummaryWriter
from torchjd.aggregation import UPGrad
from torchjd import mtl_backward
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from MyUNet import MyUNet
from Losses import PCRLoss, SegLoss, Dice, tp_fp_tn_fn, GetMetrics
from Schedulers import WarmupCosineAnnealingWithRestarts
from CustomLoader import GetDataloaders
from DataProcessing import ReconstructImageFromPatches, GetPatches
from helpers import *
from Augmenter import GetNoTransforms, GetTrainingTransforms

class MyTrainer():
    def __init__(self, nEpochs: int, modelName: str = "", tag: str = "", joint: bool = True, cat: bool = True, stopEarly: int = None, 
                 pool: bool = False, useJD: bool = False, patientDataPath="clinical_and_imaging_info.xlsx", test: bool = False):
        self.nEpochs = nEpochs
        self.stopAfterPlateauEpochs = stopEarly if stopEarly else 10_000_000
        self.joint = joint
        self.cat = cat
        self.pool = pool
        self.useJD = useJD
        
        # Parameters to change!
        self.warmup = 0.01
        self.cycles = 2
        # self.pretrainSegmentation = self.nEpochs * (1/self.cycles + self.warmup)        # pretrain for first LR annealing cycle
        self.pretrainSegmentation = 0
        self.pcrConfidence = False
        self.peakLR = 5e-5
        self.minLR = 1e-7
        self.currentEpoch = 0
        self.oversampleFG = 0.5
        self.oversampleRadius = 0.15
        self.batchSize = 10
        self.clipGrad = False
        self.downsamplePatch = 2
        self.downsampleImg = 2

        self.trainingCompose = GetTrainingTransforms()
        self.valTestCompose = GetNoTransforms()

        self.clsLosses = []
        self.PCRPercentages = []
        self.PCRBalancedAccs = []

        self.alpha = 0.99
        self.beta = 0.01

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
        nHeads = 32
        nBottleneckLayers = 8

        self.model = MyUNet(expectedPatchSize=PATCH_SIZE,
                            expectedChannels=[1, 64, 128, 256, 384, 512],   #[1, 64, 128, 256, 320, 320]
                            expectedStride=[2, 2, 2, 2, 2],
                            pretrainedDecoderPath=pretrainedDecoderPath,
                            patientDataPath=self.patientDataPath,
                            nHeads=nHeads,
                            useSkips=useSkips,
                            joint=self.joint,
                            catPosDecoder=self.cat,
                            bottleneck=bottleneck,
                            nBottleneckLayers=nBottleneckLayers,
                            useAttentionPooling=self.pool).to(device)

        dataTime = time()
        self.trDataloader, self.vlDataloader, self.tsDataloader = GetDataloaders(dataDir, self.patientDataPath, self.trainingCompose, self.valTestCompose,
                                                                                 batchSize=self.batchSize, shuffle=True, test=self.test)
        print(f"\tTook {FormatSeconds(time() - dataTime)}")

        paramGroups = [
            {'params': self.model.encoder.parameters(), 
             'lr': self.peakLR
            },
            {'params': self.model.bottleneck.parameters(), 
             'lr': 5e-6
            },
            {'params': self.model.decoder.parameters(), 
             'lr': self.peakLR
            },
            {'params': self.model.classifier.parameters(), 
             'lr': self.peakLR * 2
            },
            {'params': self.model.patientDataMod.parameters(), 
             'lr': self.peakLR * 0.5
            }
        ]
        self.optimizer = torch.optim.AdamW(paramGroups)
        self.gradScaler = torch.GradScaler(device.type)
        self.aggregator = UPGrad()

        nWarmupSteps = round(self.warmup * self.nEpochs)
        nCycleSteps = round((1 - self.warmup) * self.nEpochs / self.cycles) + 1

        self.LRScheduler = WarmupCosineAnnealingWithRestarts(
            self.optimizer,
            warmup_steps=nWarmupSteps,
            cycle_steps=nCycleSteps,
            maxLR=self.peakLR,
            minLR=self.minLR,
            damping=0.7
        )

        self.PCRloss = PCRLoss()

        # bce pos_weight should be close to the ratio of background to foreground in segmentations
        # since we oversample, we know some percentage of patches will have foreground
        # we don't know how much foreground though, since we sample randomly on the bounding box
        # so we guess that around 100x% of each oversampled patch is foreground. -> self.oversampleFG * x
        x = 0.3 * 0.6
        # If we dont oversample, then the average ratio of background to foreground is used
        # (i don't actually know this number rn, so again just guess that ~5% voxels are foreground)
        # So otherwise, posWeight = 95% / 5% = 19
        # self.bcePosWeight = min(1 / (self.oversampleFG * x) - 1, 50) if self.oversampleFG != 0 else 19
        self.bcePosWeight = 100
        self.SegLoss = SegLoss(bcePosWeight=torch.tensor([self.bcePosWeight], device=device), downsample=self.downsampleImg, 
                               normalizeTV=False, alpha=self.alpha, beta=self.beta)

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

            bceLossesThisEpoch = []
            bcePCRLossesThisEpoch = []
            bdLossesThisEpoch = []
            tvLossesThisEpoch = []
            segLossesThisEpoch = []
            pcrLossesThisEpoch = []
            rfpLossesThisEpoch = []
            diceThisEpoch = []
            truePCRs = []
            predPCRs = []

            lr = self.optimizer.param_groups[0]["lr"]
            if self.writer:
                self.writer.add_scalar("LR", lr, self.currentEpoch)

            self.model.train()

            startEpoch = time()
            nBatches = len(self.trDataloader)

            iterations = ["oversample", "oversample", "no tumor"] if not self.test else ["oversample"] 
            nHandles = len(iterations)
            for idx, struct in enumerate(self.trDataloader):        # iterate over patient cases
                mris, dmap, seg, pcr, bbox, patientIDs = zip(*struct)
                rd.shuffle(iterations)
                for i, it in enumerate(iterations):
                    if it == "oversample":
                        args = [mris, dmap, seg, PATCH_SIZE * self.downsamplePatch, NUM_PATCHES, self.oversampleFG,
                                self.oversampleRadius, bbox, self.downsampleImg, False]
                    else:
                        args = [mris, dmap, seg, PATCH_SIZE * self.downsamplePatch, NUM_PATCHES, -1, 0, bbox, self.downsampleImg, False]
                    phases, distMap, target, patchIndices = GetPatches(*args)
                    phases: torch.Tensor    = phases.transpose(1, 2).to(self.device, non_blocking=True)
                    distMap: torch.Tensor   = distMap.to(self.device, non_blocking=True)
                    target: torch.Tensor    = target.to(self.device, non_blocking=True)
                    patchIndices            = patchIndices.to(self.device)

                    phases = DownsampleTensor(phases, PATCH_SIZE)

                    with torch.autocast(self.device.type):
                        segOut, sharedFeatures, pcrOut, rfpLoss = self.model(phases, patientIDs, patchIndices)
                        loss = 0
                        pcrLoss = None
                        if self.currentEpoch >= self.pretrainSegmentation and pcrOut is not None and self.joint:
                            pcrLoss: torch.Tensor = self.PCRloss(pcrOut, pcr)
                            loss = pcrLoss + rfpLoss if pcrLoss is not None else rfpLoss
                            truePCRs.append(torch.stack(pcr))
                            predPCRs.append(pcrOut)

                            if pcrLoss is not None:
                                bcePCRLossesThisEpoch.append(pcrLoss.item())

                            pcrLossesThisEpoch.append(loss.item())
                            rfpLossesThisEpoch.append(rfpLoss.item())

                        if segOut is not None:
                            segOut = UpsampleTensor(segOut, PATCH_SIZE * self.downsamplePatch)
                            segLoss: torch.Tensor = self.SegLoss(segOut, target, distMap)

                            bceLoss = self.SegLoss.bc * self.SegLoss.BCWeight
                            bdLoss = self.SegLoss.bd * self.SegLoss.BDWeight
                            tvLoss = self.SegLoss.tv * self.SegLoss.TVWeight

                            segLossesThisEpoch.append(segLoss.item())
                            bceLossesThisEpoch.append(bceLoss.item())
                            tvLossesThisEpoch.append(tvLoss.item())
                            bdLossesThisEpoch.append(bdLoss.item())
                            
                            # now only do foreground for "real" Dice score
                            segOut: torch.Tensor = (segOut > 0).int()
                            dice = Dice(segOut.detach().cpu(), target.detach().cpu())
                            diceThisEpoch.append(dice)

                            totalLoss = loss + segLoss

                            print(f"\tTraining Batch {idx + (1 + i) / nHandles:.2f}/{nBatches:.2f}: {totalLoss:.4f} = BCE Loss: {bceLoss:.4f} + BD Loss: {bdLoss:.4f}{f" + PCR Loss {pcrLoss:.4f} + RFP Loss {rfpLoss:.4f}" if pcrLoss is not None else ""}", end='\r')
                        else:
                            print(f"\tTraining Batch {idx + (1 + i) / nHandles:.2f}/{nBatches:.2f}: {loss:.4f} = PCR Loss {pcrLoss:.4f} + RFP Loss {rfpLoss:.4f}", end='\r')

                    del phases, distMap, patchIndices

                    # free as much space as possible before backward()
                    del segOut, pcrOut, target
                    
                    self.optimizer.zero_grad()
                    if self.joint and pcrLoss is not None and not pcrLoss.isnan().any():
                        if self.useJD:
                            assert sharedFeatures is not None, "Cannot do joint backward without shared features!"

                            losses = self.gradScaler.scale([segLoss, loss])
                            mtl_backward(losses=losses, 
                                        features=sharedFeatures, 
                                        aggregator=self.aggregator,
                                        tasks_params=[list(self.model.decoder.parameters()), 
                                                      list(self.model.classifier.parameters()) + 
                                                      list(self.model.patientDataMod.parameters())],
                                        shared_params=list(self.model.encoder.parameters()) + 
                                                      list(self.model.bottleneck.parameters()))
                        else:
                            scaledLoss = self.gradScaler.scale(totalLoss)
                            scaledLoss.backward()
                    elif pcrLoss is not None:
                        scaledLoss = self.gradScaler.scale(loss)
                        scaledLoss.backward()
                    else:
                        scaledLoss = self.gradScaler.scale(segLoss)
                        scaledLoss.backward()
                    
                    if self.clipGrad:
                        self.gradScaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                    
                    self.gradScaler.step(self.optimizer)
                    self.gradScaler.update()

                    del loss, segLoss, pcrLoss, sharedFeatures
                del mris, dmap, seg, pcr
            torch.cuda.empty_cache()

            if self.joint:
                predPCRs = torch.cat(predPCRs).detach().squeeze().float().cpu().numpy()
                truePCRs = torch.cat(truePCRs).int().cpu().numpy()

                # Mask out invalid entries
                mask = truePCRs != -1
                truePCRs_masked = truePCRs[mask]
                predPCRs_masked = predPCRs[mask]

                # Now compute AUC only on valid entries
                fpr, sens, _ = roc_curve(truePCRs_masked, predPCRs_masked)
                spec = 1 - fpr
                i = np.argmax(spec + sens - 1)
                trSens = sens[i]
                trSpec = spec[i]
                trAuc = roc_auc_score(truePCRs_masked, predPCRs_masked)
                trPrauc = average_precision_score(truePCRs_masked, predPCRs_masked)
            print()
            if self.writer:
                self.writer.add_scalars('Seg Loss/Training', {"Overall": Mean(segLossesThisEpoch), 
                                                              "Boundary": Mean(bdLossesThisEpoch),
                                                              "BCE": Mean(bceLossesThisEpoch),
                                                              "Tversky": Mean(tvLossesThisEpoch)}, self.currentEpoch)
                
                if len(pcrLossesThisEpoch) > 0:
                    self.writer.add_scalars(f"PCR Loss/Training", {"Overall": Mean(pcrLossesThisEpoch),
                                                                   "RFP": Mean(rfpLossesThisEpoch),
                                                                   "BCE": Mean(bcePCRLossesThisEpoch)}, self.currentEpoch)

                if self.currentEpoch % 10 == 0 and self.logGradients:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and any(k in name for k in ["encoder", "bottleneck", "decoder", "classifier", "patientData"]):
                            self.writer.add_histogram(f'Training Gradients/{name}', param.grad, self.currentEpoch)

            print(f"\tTraining loop took {FormatSeconds(time() - startEpoch)}")
            
            # =========================================
            #              VALIDATION LOOP
            # =========================================

            segLossesVal = []
            bceLossesVal = []
            bcePCRLossesVal = []
            bdLossesVal = []
            tvLossesVal = []
            pcrLossesVal = []
            rfpLossesVal = []
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
                truePCRs.append(pcr)
                phases, distMap, target, patchIndices = GetPatches(mris, dmap, seg, PATCH_SIZE * self.downsamplePatch, 
                                                                   NUM_PATCHES, 0, 0, bbox, self.downsampleImg, True)
                phases: torch.Tensor    = phases.transpose(1, 2).to(self.device, non_blocking=True)
                distMap: torch.Tensor   = distMap.to(self.device, non_blocking=True)
                target: torch.Tensor    = target.to(self.device, non_blocking=True)
                patchIndices            = patchIndices.to(self.device)

                phases = DownsampleTensor(phases, PATCH_SIZE)

                with torch.no_grad(), torch.autocast(self.device.type):
                    n = patchIndices.shape[1]
                    allOuts = []
                    for startI in range(0, n, CHUNK_SIZE):
                        stopI = min(startI + CHUNK_SIZE, n)
                        out = self.model(phases[:, :, startI:stopI], patientIDs, patchIndices[:, startI:stopI])
                        allOuts.append(out)
                        del out

                    segOuts, _, pcrOuts, rfpLosses = zip(*allOuts)
                    if self.joint:
                        pcrOut = torch.cat(pcrOuts, dim=1).mean()
                        pcrLoss = self.PCRloss(pcrOut.unsqueeze(-1), pcr)
                        rfpLoss = torch.stack(rfpLosses).mean()
                        loss: torch.Tensor = pcrLoss + rfpLoss if pcrLoss is not None else rfpLoss
                        predPCRs.append(pcrOut.mean())

                        if pcrLoss is not None:
                            bcePCRLossesVal.append(pcrLoss.item())

                        pcrLossesVal.append(loss.item())
                        rfpLossesVal.append(rfpLoss.item())

                        segOut = torch.cat(segOuts, dim=1)
                        segOut = UpsampleTensor(segOut, PATCH_SIZE * self.downsamplePatch)
                        segLoss = self.SegLoss(segOut.float(), target, distMap)

                        bceLoss = self.SegLoss.bc * self.SegLoss.BCWeight
                        bdLoss = self.SegLoss.bd * self.SegLoss.BDWeight
                        tvLoss = self.SegLoss.tv * self.SegLoss.TVWeight

                        segLossesVal.append(segLoss.item())
                        bceLossesVal.append(bceLoss.item())
                        tvLossesVal.append(tvLoss.item())
                        bdLossesVal.append(bdLoss.item())

                        totalLoss = loss + segLoss

                        print(f"\tValidation Batch {idx+1}/{nBatches}: {totalLoss:.4f} = BCE Loss: {bceLoss:.4f} + BD Loss: {bdLoss:.4f}{f" + PCR Loss {pcrLoss:.4f} + RFP Loss {rfpLoss:.4f}" if pcrLoss is not None else ""}", end='\r')
                    else:
                        raise Exception("Not implemented!")

                del phases, distMap
                
                patchIndices = patchIndices.cpu()
                segOut: torch.Tensor = (segOut > 0).int().cpu()
                target = target.int().cpu()
                dicePatches = []
                diceFull = []
                sens = []
                spec = []
                for i in range(len(patientIDs)):
                    dicePatches.append(Dice(segOut[i], target[i]))

                    segImageArr     = ReconstructImageFromPatches(segOut[i], patchIndices[i], PATCH_SIZE * self.downsamplePatch)
                    targetImageArr  = ReconstructImageFromPatches(target[i], patchIndices[i], PATCH_SIZE * self.downsamplePatch)
                    
                    diceFull.append(Dice(segImageArr, targetImageArr))
                    tp, fp, tn, fn = tp_fp_tn_fn(segImageArr, targetImageArr)
                    sens.append(tp / (tp + fn))
                    spec.append(tn / (tn + fp))

                diceValFull.append(Mean(diceFull))
                diceValPatches.append(Mean(dicePatches))
                sensVal.append(Mean(sens))
                specVal.append(Mean(spec))

                del segOut, pcrOut, target, patchIndices, segLoss, pcrLoss, mris, dmap, seg, pcr
            torch.cuda.empty_cache()
            
            self.LRScheduler.step()
            print()

            if self.joint and self.currentEpoch >= self.pretrainSegmentation:
                predPCRs = torch.stack(predPCRs).float().cpu().numpy()
                truePCRs = np.array(truePCRs).astype(int).squeeze()

                # Mask out invalid entries
                mask = truePCRs != -1
                truePCRs_masked = truePCRs[mask]
                predPCRs_masked = predPCRs[mask]

                # Now compute AUC only on valid entries
                fpr, sens, _ = roc_curve(truePCRs_masked, predPCRs_masked)
                spec = 1 - fpr
                i = np.argmax(spec + sens - 1)
                vlSens = sens[i]
                vlSpec = spec[i]
                vlAuc = roc_auc_score(truePCRs_masked, predPCRs_masked)
                vlPrauc = average_precision_score(truePCRs_masked, predPCRs_masked)

            # MODEL CHECKPOINTING
            if self.writer:
                avgSegValLoss = Mean(segLossesVal)
                
                self.writer.add_scalars('Seg Loss/Validation', {"Overall": avgSegValLoss, 
                                                              "Boundary": Mean(bdLossesVal),
                                                              "BCE": Mean(bceLossesVal),
                                                              "Tversky": Mean(tvLossesVal)}, self.currentEpoch)

                avgDiceVal = Mean(diceValFull)
                self.writer.add_scalars('Dice', {"Train": Mean(diceThisEpoch), 
                                                 "Val - Full": avgDiceVal,
                                                 "Val - Patches": Mean(diceValPatches)}, self.currentEpoch)

                self.writer.add_scalars('Val Metrics', {"Sensitivity": Mean(sensVal),
                                                        "Specificity": Mean(specVal)}, self.currentEpoch)

                # if avgDiceVal > bestSeg:
                #     bestSeg = avgDiceVal
                #     bestSegEpoch = self.currentEpoch
                #     print(f"Saving Best Seg: epoch {self.currentEpoch}")
                #     self.saveModel(f"BestSeg{self.tag}")
                
                if len(pcrLossesVal) > 0:
                    avgPCRValLoss = Mean(pcrLossesVal)
                    self.writer.add_scalars("PCR Loss/Validation", {"Overall": avgPCRValLoss,
                                                                     "RFP": Mean(rfpLossesVal),
                                                                     "BCE": Mean(bceLossesVal)}, self.currentEpoch)
                    self.writer.add_scalars("PCR AUC", {"Train": trAuc, "Val": vlAuc}, self.currentEpoch)
                    self.writer.add_scalars("PCR PR AUC", {"Train": trPrauc, "Val": vlPrauc}, self.currentEpoch)
                    self.writer.add_scalars("Sensitivity", {"Train": trSens, "Val": vlSens}, self.currentEpoch)
                    self.writer.add_scalars("Specificity", {"Train": trSpec, "Val": vlSpec}, self.currentEpoch)

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
        
        self.model.ret = "all"

        resultsFolder = f"outputs{self.tag}{resultsTag}"
        outputPath = os.path.join(self.outputFolder, resultsFolder, outputPath)
        outputPathPCR = os.path.join(self.outputFolder, resultsFolder, outputPathPCR)
        os.makedirs(outputPath, exist_ok=True)
        os.makedirs(outputPathPCR, exist_ok=True)

        import pandas as pd
        from MAMAMIA.src.challenge.metrics import hausdorff_distance
        
        print("Running inference!")
        scoreDF = None

        truePCRs = []
        predPCRs = []

        for struct in self.tsDataloader:
            phases, dmap, seg, pcr, bbox, patientIDs = zip(*struct)
            phases, _, target, patchIndices = GetPatches(phases, dmap, seg, PATCH_SIZE * self.downsamplePatch, 
                                                         NUM_PATCHES, 0, 0, bbox, self.downsampleImg, True)
            phases: torch.Tensor    = phases.transpose(1, 2).to(self.device, dtype=DTYPE_PHASE, non_blocking=True)
            phase1: torch.Tensor    = phases[:, 1].float().detach().cpu()
            patchIndices            = patchIndices.to(self.device, non_blocking=True)
            target: torch.Tensor    = target.to(self.device, non_blocking=True)

            phases = DownsampleTensor(phases, PATCH_SIZE)
            truePCRs.append(pcr)
            predsThisBatch = []
            with torch.no_grad(), torch.autocast(self.device.type):
                n = patchIndices.shape[1]
                allOuts = []
                for startI in range(0, n, CHUNK_SIZE):
                    stopI = min(startI + CHUNK_SIZE, n)
                    out = self.model(phases[:, :, startI:stopI], patientIDs, patchIndices[:, startI:stopI], DownsampleTensor(target.float(), PATCH_SIZE)[:, startI:stopI])
                    allOuts.append(out)    
                    del out
                segOuts, _, pcrOuts, _ = zip(*allOuts)
                if self.joint:
                    pcrOut = torch.cat(pcrOuts, dim=1)
                    predPCRs.append(pcrOut.float().mean().item())
                    predsThisBatch.append(pcrOut.float().mean().item())

                    segOut = torch.cat(segOuts, dim=1)
                    segOut = UpsampleTensor(segOut, PATCH_SIZE * self.downsamplePatch)
            del phases

            if None not in pcrOuts:
                for i, patientID in enumerate(patientIDs):
                    row = {"Patient ID": [patientID], "PCR": [pcr[i].item()], "Pred PCR": [predsThisBatch[i]]}
                        
                    row = pd.DataFrame(row)
                    if scoreDF is not None:
                        scoreDF = pd.concat([scoreDF, row])
                    else:
                        scoreDF = row

                    print(" " * 80, end="\r")
                    print(f"Finished patient {patientID}\tPCR: {pcr[i].item()}\tPred: {predsThisBatch[i]:.4f}", end="\r")

            if None not in segOuts:
                patchIndices = patchIndices.detach().cpu()
                for i, patientID in enumerate(patientIDs):
                    dicePatches = Dice(segOut[i], target[i])
                    phaseImageArr   = ReconstructImageFromPatches(phase1[i], patchIndices[i], PATCH_SIZE * self.downsamplePatch)
                    segImageArr     = ReconstructImageFromPatches(segOut[i], patchIndices[i], PATCH_SIZE * self.downsamplePatch)
                    targetImageArr  = ReconstructImageFromPatches(target[i], patchIndices[i], PATCH_SIZE * self.downsamplePatch)
                    
                    dice = Dice(segImageArr, targetImageArr)
                    segMetrics = GetMetrics(*tp_fp_tn_fn(segImageArr, targetImageArr), "Seg")

                    segImageArr = segImageArr.numpy()
                    targetImageArr = targetImageArr.numpy()

                    hausdorff = min(hausdorff_distance(targetImageArr, segImageArr), 1000)

                    row = {"Patient ID": [patientID], "Dice (Full Image)": [dice], "Dice (Avg Over Patches)": [dicePatches],"HD95": [hausdorff]} | segMetrics
                    
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
        auc = None
        if self.joint:
            predPCRs = np.array(predPCRs)
            truePCRs = np.array(truePCRs).astype(int).squeeze()

            # Mask out invalid entries
            mask = truePCRs != -1
            truePCRs_masked = truePCRs[mask]
            predPCRs_masked = predPCRs[mask]

            # Now compute AUC only on valid entries
            auc = roc_auc_score(truePCRs_masked, predPCRs_masked)
            prauc = average_precision_score(truePCRs_masked, predPCRs_masked)
            print("===================")
            print("TESTING AUC:", auc)
            print("TESTING PRAUC:", prauc)
            print("===================")
            scoreDF["ROC AUC"] = auc
            scoreDF["PR AUC"] = prauc
        savePath = os.path.join(self.outputFolder, resultsFolder, "scores.csv")
        scoreDF.to_csv(savePath, index=False)

        print(f"Saved results to: {savePath}")
        metrics = [col for col in scoreDF.columns if col != "Patient ID" and "PCR" not in col]
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
    # pretrainedDecoderPath = r"transformerResults\TransformerTSJointWithSkips\BestSegOct20-DownsampleImages.pth"
    pretrainedDecoderPath = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cat = 128
    pool = True
    tag = f"Jan26-{f'Cat{cat}' if cat is not None else 'Add'}{'Pool' if pool else 'Cls'}MoreClassifierDropoutAndSmallerPatientFeatureEmbedding"
    # tag = "Oct24-DownsampleImagesWithPCR"
    bottleneck = BOTTLENECK_SPATIOTEMPORAL
    # bottleneck = BOTTLENECK_TRANSFORMERTS
    # bottleneck = BOTTLENECK_TRANSFORMERST
    # bottleneck = BOTTLENECK_CONV
    skips = True
    joint = True
    test  = False        # testing the model on a few specific patients so we don't have to wait for the dataloader
    modelName = f"{bottleneck}{"Joint" if joint else ""}{"With" if skips else "No"}Skips" #{"-TEST" if test else ""}"
    trainer = MyTrainer(nEpochs=800, modelName=modelName, tag=tag, joint=joint, cat=cat, pool=pool, useJD=False, test=test)
    
    trainer.setup(dataDir, 
                  device, 
                  pretrainedDecoderPath=pretrainedDecoderPath, 
                  useSkips=skips, 
                  bottleneck=bottleneck)
    print(f"Set up model {modelName}")

    # trainer.train(continueTraining=True, modelName=f"Latest{tag}.pth")
    trainer.train()
    trainer.inference(f"Latest{tag}.pth", "Latest")
    if joint:
        trainer.inference(f"BestPCR{tag}.pth", "BestPCR")
    else:
        trainer.inference(f"BestSeg{tag}.pth", "BestSeg")
