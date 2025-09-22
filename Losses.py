import torch, torch.nn as nn
from helpers import EPS

class PCRLoss(nn.Module):
    def __init__(self):
        super(PCRLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.4))  # this data set sees 70% negative, 30% positive

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # Convert list to tensor if needed
        if isinstance(targets, list):
            targets = torch.tensor(targets, device=logits.device)

        targets = targets.float()

        # Ensure logits and targets match in shape
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)  # shape (B,)

        loss = self.bce(logits.float(), targets)

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
        sumTarget = target.sum(axes)
        sumInput  = x.sum(axes)
        intersect = (x * target).sum(axes)

        dice    = (intersect * 2 + EPS) / (sumInput + sumTarget + EPS)
        fgMask  = (sumTarget > 0).float()
        bgMask  = -fgMask + 1
        
        # weighting shenanigans. Balance foreground vs background
        fgW     = bgMask.mean()
        bgW     = fgMask.mean()

        weighted = (fgMask * fgW + bgMask * bgW) * dice
        return 1 - weighted.mean()
    
class SegLoss(nn.Module):
    def __init__(self, bcePosWeight: float):
        super().__init__()

        self.DiceWeight = 1
        self.BCEWeight  = 1
        self.BDWeight   = 3e-2

        self.DiceLoss   = DiceLoss()
        self.BCELoss    = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bcePosWeight))
        self.BDLoss     = BoundaryLoss(class_idx=0)

    # expect x shape: [B, N, 1, X, Y, Z]
    def forward(self, x: torch.Tensor, target: torch.Tensor, dmaps: torch.Tensor) -> tuple[torch.Tensor]:
        bceLoss: torch.Tensor   = self.BCELoss(x, target.float())
        
        # now we activate with sigmoid since these losses assume prob input
        x = torch.sigmoid(x)
        
        # oldDiceLoss = self.OldDiceLoss(x, target)
        diceLoss: torch.Tensor  = self.DiceLoss(x, target)
        bdLoss: torch.Tensor    = self.BDLoss(x, dmaps)
        return bceLoss * self.BCEWeight, diceLoss * self.DiceWeight, (1 + bdLoss) * self.BDWeight

# Assume input and target are same size (X, Y, Z) and the values of each entry are binary.
# Return Dice loss and Dice score
def Dice(recon: torch.Tensor, target: torch.Tensor):
    r: torch.Tensor = recon.to(dtype=bool)
    t: torch.Tensor = target.to(dtype=bool)

    tp_2  = int(torch.sum(r & t).item()) << 1   # true positives
    mag_r = int(torch.sum(r).item())            # magnitude of recon
    mag_t = int(torch.sum(t).item())            # magnitude of target
    diceScore = (tp_2 + EPS) / (mag_r + mag_t + EPS)
    return diceScore
