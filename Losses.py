import torch, torch.nn as nn

from MAMAMIA.nnUNet.nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

class PCRLoss(nn.Module):
    def __init__(self):
        """
        alpha and beta are weighting terms in case you want to combine BCE with another loss
        for example: total_loss = alpha * BCE + beta * focal or another auxiliary term.
        """
        super(PCRLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.4))  # this data set sees 70% negative, 30% positive

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
            logits = logits.squeeze(1)  # shape (B,)

        loss = self.bce(logits.float(), targets)

        return loss
    
class BoundaryLoss(nn.Module):
    """
    A simple implementation of Boundary Loss for binary segmentation.

    This loss function encourages the model to accurately predict the boundaries
    of objects by penalizing discrepancies between the predicted segmentation
    and the ground truth signed distance map (SDM).

    The core idea is based on the formulation: L_BD = sum(predicted_probability * ground_truth_SDM)
    where SDM is negative inside the object, positive outside, and zero at the boundary.
    Minimizing this sum encourages high probabilities where SDM is negative (inside GT)
    and low probabilities where SDM is positive (outside GT).

    Args:
        sigmoid (bool): If True, applies a sigmoid activation to the inputs.
                        Set to True if your model outputs logits.
                        Set to False if your model outputs probabilities (0-1).
        reduction (str): Specifies the reduction to apply to the output:
                         'mean' | 'sum' | 'none'. Default: 'mean'.
                         'mean' will average the loss over the batch.
        class_idx (int, optional): The start index of the foreground class in the target tensor.
                                   Defaults to 1 (assuming channel 0 is bg, channel 1 is fg).
    """
    def __init__(self, reduction: str = 'mean', class_idx: int = 1):
        super(BoundaryLoss, self).__init__()
        self.reduction = reduction
        self.class_idx = class_idx

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction must be 'mean', 'sum', or 'none', but got {reduction}")

    def forward(self, inputs: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Boundary Loss.

        Args:
            inputs (torch.Tensor): Activated outputs from the model.
                                   Shape: (N, C, H, W, D) for 3D or (N, C, H, W) for 2D.
                                   Expect C = 2 for binary background vs foreground
            dist_maps (torch.Tensor): Distance maps for ground truth segmentations.
                                    Shape: (N, C, H, W, D) or (N, C, H, W).

        Returns:
            torch.Tensor: The calculated Boundary Loss.
        """
        inputs = inputs[:, self.class_idx:, ...]    # select foreground only
        # print(f"Inputs have nan after sigmoid: {torch.any(inputs.isnan())}")
        # print(f"Inputs after sigmoid: {inputs.min(), inputs.max()}")

        # Calculate the core Boundary Loss term
        # Element-wise product of predicted probabilities and SDM
        loss: torch.Tensor = inputs * dist_maps
        # print("Boundary Loss has gradient:", loss.requires_grad)
        # print(f"inputs: {inputs.shape}, {inputs.dtype}")
        # print(f"Dist maps: {dist_maps.shape}, {dist_maps.dtype}")
        # print(f"\t{dist_maps.min(), dist_maps.max()}")
        # print(f"BD loss: {loss.shape}")
        # print()

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss
    
class SegLoss(nn.Module):
    def __init__(self, bcePosWeight: float):
        super().__init__()

        softDiceKWArgs = {'batch_dice': False,      # we will likely use a batch size of 1
                          'do_bg': True,            # our segmentation will just be one channel
                          'smooth': 1e-5, 
                          'ddp': False}
        self.DiceWeight = 1
        self.BCEWeight  = 1
        self.BDWeight   = 1

        self.DiceLoss   = MemoryEfficientSoftDiceLoss(**softDiceKWArgs)
        self.BCELoss    = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bcePosWeight))
        self.BDLoss     = BoundaryLoss()

    def forward(self, x: torch.Tensor, target: torch.Tensor, dmaps: torch.Tensor):
        bceLoss  = self.BCELoss(x, target.float())
        
        # now we activate with softmax since these losses assume prob input
        x = torch.softmax(x, dim=-4)

        diceLoss = self.DiceLoss(x, target)
        bdLoss   = self.BDLoss(x, dmaps)
        return bceLoss * self.BCEWeight, diceLoss * self.DiceWeight, bdLoss * self.BDWeight

# Assume input and target are same size (X, Y, Z) and the values of each entry are binary.
# Return Dice loss and Dice score
def Dice(recon: torch.Tensor, target: torch.Tensor):
    r: torch.Tensor = recon.to(dtype=bool)
    t: torch.Tensor = target.to(dtype=bool)

    smooth = 1e-4

    tp_2  = int(torch.sum(r & t).item()) << 1   # true positives
    mag_r = int(torch.sum(r).item())            # magnitude of recon
    mag_t = int(torch.sum(t).item())            # magnitude of target
    diceScore = (tp_2 + smooth) / (mag_r + mag_t + smooth)
    return diceScore
