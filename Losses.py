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
        # self.classWeights = torch.tensor([0.1, 0.9])

    # assume x is activated with sigmoid, foreground probs only
    # [B, N, X, Y, Z],   B = numBatches, N = numPatches, X,Y,Z = patchSize
    def forward(self, x: torch.Tensor, target: torch.Tensor):
        # extract foreground vs background predictions
        x = x.unsqueeze(2)
        target = target.unsqueeze(2)

        x = torch.cat([1-x, x], dim=2)
        target = torch.cat([1-target, target], dim=2)

        axes = tuple(range(x.ndim - 3, x.ndim))

        sumTarget = target.sum(axes)
        sumInput  = x.sum(axes)
        intersect = (x * target).sum(axes)

        N = x.numel()
        dice    = (intersect * 2 / N + EPS) / ((sumInput + sumTarget) / N + EPS)

        # Balance foreground vs background
        classFreq = sumTarget.mean(dim=(0, 1))  # mean over batch and patch dims
        classWeights = 1 - (classFreq / (classFreq.sum()))
        classWeights = classWeights / (classWeights.sum())
        
        weighted = (dice * classWeights.view(1, 1, -1)).sum(dim=2)

        return 1 - weighted.mean()
    
class TverskyLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, normalize: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.normalize = normalize

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        axes = tuple(range(1, x.ndim))
        TP = (x * target).sum(axes)
        FP = (x * (1 - target)).sum(axes)
        FN = ((1 - x) * target).sum(axes)

        if self.normalize:
            n = x.numel()
            TP = TP / n
            FP = FP / n
            FN = FN / n
        
        tversky = (TP + EPS) / (TP + self.alpha*FN + self.beta*FP + EPS)

        return 1 - tversky.mean()
    
class SegLoss(nn.Module):
    def __init__(self, bcePosWeight: torch.Tensor, downsample: int, normalizeTV: bool, 
                 alpha: float = 0.5, beta: float = 0.5):
        super().__init__()

        self.BCWeight  = 1
        self.BDWeight  = 1 / downsample
        self.TVWeight  = 2

        self.BCLoss = nn.BCEWithLogitsLoss(pos_weight=bcePosWeight)
        self.BDLoss = BoundaryLoss(class_idx=0)
        self.TVLoss = TverskyLoss(alpha=alpha, beta=beta, normalize=normalizeTV)

    # expect x shape: [B, N, 1, X, Y, Z]
    def forward(self, x: torch.Tensor, target: torch.Tensor, dmaps: torch.Tensor) -> tuple[torch.Tensor]:
        self.bc: torch.Tensor = self.BCLoss(x, target.float())
        
        # now we activate with sigmoid since these losses assume prob input
        x = torch.sigmoid(x)
        self.bd: torch.Tensor = self.BDLoss(x, dmaps)
        self.tv: torch.Tensor = self.TVLoss(x, target)

        return self.tv * self.TVWeight + self.bc * self.BCWeight + self.bd * self.BDWeight

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