import torch, torch.nn as nn
from helpers import EPS

class PCRLoss(nn.Module):
    def __init__(self):
        super(PCRLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.4))  # this data set sees 70% negative, 30% positive

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # Convert list to tensor if needed
        if isinstance(targets, list) or isinstance(targets, tuple):
            targets = torch.tensor(targets, device=logits.device)

        # ignore missing pcr values
        mask = targets != -1

        # Ensure logits and targets match in shape
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)  # shape (B,)

        loss: torch.Tensor = self.bce(logits[mask].float(), targets[mask].float())

        return loss
    
class PCRLossWithConfidence(nn.Module):
    def __init__(self, pos_weight=2.4):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight))

    def forward(self, logits: tuple[torch.Tensor], targets: torch.Tensor):
        # logits, confProb: (B,1), targets: (B,)
        logits, confProb = logits[0], logits[1]
        if isinstance(targets, (list, tuple)):
            targets = torch.tensor(targets, device=logits.device)

        mask = targets != -1
        logits = logits[mask].float()
        confProb = confProb[mask].float()
        targets = targets[mask].float()

        # BCE per chunk
        lossPerChunk = self.bce(logits, targets)

        # Confidence weights (0-1)
        weights = confProb.squeeze(1)

        # Weighted mean
        loss = (lossPerChunk * weights).sum() / (weights.sum() + 1e-6)
        return loss
    
class BoundaryLoss(nn.Module):
    def __init__(self, reduction: str = 'mean', class_idx: int = 1):
        super(BoundaryLoss, self).__init__()
        self.reduction = reduction
        self.class_idx = class_idx

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction must be 'mean', 'sum', or 'none', but got {reduction}")

    # Assume input shape: (N, C, H, W, D)
    def forward(self, inputs: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        inputs = inputs[:, self.class_idx:, ...]    # select foreground only

        # Calculate the core Boundary Loss term
        # Element-wise product of predicted probabilities and SDM
        loss: torch.Tensor = inputs * dist_maps

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss
        
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # assume x is activated with sigmoid, foreground probs only
    # [B, N, C, X, Y, Z],   B = numBatches, N = numPatches, C = 1, X,Y,Z = patchSize
    def forward(self, x: torch.Tensor, target: torch.Tensor):
        axes = tuple(range(x.ndim - 3, x.ndim))
        # TESTING: use mean instead of sum. no Dice exclusion.
        sumTarget = target.mean(axes)

        # don't do Dice backprop when no foreground, let other losses handle that
        # if sumTarget.sum().item() == 0:
        #     return torch.tensor(1., device=x.device)
        
        sumInput  = x.mean(axes)
        intersect = (x * target).mean(axes)

        dice    = (intersect * 2 + EPS) / (sumInput + sumTarget + EPS)

        # Balance foreground vs background
        fgMask  = (sumTarget > 0).float()
        bgMask  = -fgMask + 1
        
        # If p% of patches contain tumor, scale those patches by q% = 100% - p%
        # Scale the rest of the patches by p%. Then normalize.
        fgW = bgMask.mean()
        bgW = fgMask.mean()

        # Using fgMask.mean() as a proxy for Pr(Patch contains foreground) = p%
        # Because we are scaling only the patches in fgMask by q% 
        # E[Weighted Dice] = E[(p% * q% + q% * p%) * Dice]
        #                  = fgW * fgMask.mean() * E[Dice] + bgW * bgMask.mean() * E[Dice] 
        #                  = E[Dice] * 2 * fgW * bgW
        normFactor = (1 / (fgW * bgW * 2)) if fgW.item() != 0 and bgW.item() != 0 else 1

        weighted = (fgMask * fgW + bgMask * bgW) * dice * normFactor
        return 1 - weighted.mean()
    
class SegLoss(nn.Module):
    def __init__(self, bcePosWeight: float, downsample: int):
        super().__init__()

        self.DiceWeight = 3
        self.BCEWeight  = 2
        self.BDWeight   = 1 / downsample

        self.DiceLoss   = DiceLoss()
        self.BCELoss    = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bcePosWeight))
        self.BDLoss     = BoundaryLoss(class_idx=0)

    # expect x shape: [B, N, 1, X, Y, Z]
    def forward(self, x: torch.Tensor, target: torch.Tensor, dmaps: torch.Tensor) -> tuple[torch.Tensor]:
        bceLoss: torch.Tensor = self.BCELoss(x, target.float())
        
        # now we activate with sigmoid since these losses assume prob input
        x = torch.sigmoid(x)
        
        # oldDiceLoss = self.OldDiceLoss(x, target)
        diceLoss: torch.Tensor  = self.DiceLoss(x, target)
        bdLoss: torch.Tensor    = self.BDLoss(x, dmaps)
        return bceLoss * self.BCEWeight, diceLoss * self.DiceWeight, (1 + bdLoss) * self.BDWeight

# Assume input and target are same size (X, Y, Z) and the values of each entry are binary.
# Return Dice loss and Dice score
def Dice(recon: torch.Tensor, target: torch.Tensor):
    r: torch.Tensor = recon.bool()
    t: torch.Tensor = target.bool()

    tp_2  = int(torch.sum(r & t).item()) << 1   # true positives
    mag_r = int(torch.sum(r).item())            # magnitude of recon
    mag_t = int(torch.sum(t).item())            # magnitude of target
    diceScore = (tp_2 + EPS) / (mag_r + mag_t + EPS)
    return diceScore

def tp_fp_tn_fn(recon: torch.Tensor, target: torch.Tensor):
    r: torch.Tensor = recon.bool()
    t: torch.Tensor = target.bool()

    tp = int(torch.sum(r & t).item())
    fp = int(torch.sum(r & ~t).item())
    tn = int(torch.sum(~r & ~t).item())
    fn = int(torch.sum(~r & t).item())

    return tp, fp, tn, fn

def GetMetrics(tp, fp, tn, fn, suffix):
    precision = tp / max(1, tp + fp)                # positive predictive value
    recall = tp / max(1, tp + fn)                   # true positive rate / sensitivity
    specificity = tn / max(1, tn + fp)              # true negative rate

    f1 = 2 * precision * recall / max(1e-10, precision + recall)
    iou = tp / max(1, tp + fp + fn)                 # intersection over union
    acc = (tp + tn) / max(1, tp + tn + fp + fn)     # accuracy
    bal_acc = (recall + specificity) / 2            # balanced accuracy
    npv = tn / max(1, tn + fn)                      # negative predictive value
    fpr = fp / max(1, fp + tn)                      # false positive rate
    fnr = fn / max(1, fn + tp)                      # false negative rate

    row = {
        "TP": [tp], "FP": [fp], "TN": [tn], "FN": [fn],
        "Sens": [recall], "Spec": [specificity],
        "Acc": [acc],
        "Prec": [precision],
        "F1": [f1],
        "IoU": [iou],
        "Bal Acc": [bal_acc],
        "NPV": [npv],
        "FPR": [fpr],
        "FNR": [fnr]
    }
    row = {k + f" ({suffix})": v for k, v in row.items()}
    return row