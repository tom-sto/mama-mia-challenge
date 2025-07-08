import torch, torch.nn as nn
from typing import Union

class PCRLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        """
        alpha and beta are weighting terms in case you want to combine BCE with another loss
        for example: total_loss = alpha * BCE + beta * focal or another auxiliary term.
        """
        super(PCRLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta  # currently unused, but kept for extension
        self.bce = nn.BCEWithLogitsLoss()  # handles sigmoid internally

    def forward(self, logits: torch.Tensor, targets: Union[list[int], torch.Tensor]):
        """
        Args:
            logits: (B,) or (B, 1) raw model outputs (before sigmoid)
            targets: list of ints (0 or 1), or a torch.Tensor of shape (B,)
        """
        # Convert list to tensor if needed
        if isinstance(targets, list):
            targets = torch.tensor(targets, dtype=torch.float32, device=logits.device)

        # Ensure logits and targets match in shape
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)  # shape (B,)

        # Filter out entries where targets are -1
        mask = targets != -1
        logits = logits[mask]
        targets = targets[mask]
        print("\tLogits:", logits)
        print("\tTargets:", targets)
        loss = self.alpha * self.bce(logits, targets)

        return loss
    

if __name__ == "__main__":
    l = PCRLoss()
    logits = torch.randn(10, 1)
    targets = [0,1,1,1,0,1,0,0,0,-1]
    print(logits)
    print(targets)
    print(l(logits, targets))
    print(l(torch.tensor([0.]), [0]))
    print(l(torch.tensor([-10.]), [1]))
    print(l(torch.tensor([-10.]), [-1]))