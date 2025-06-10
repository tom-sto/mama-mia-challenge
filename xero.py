import torch
import torch.nn as nn
import torch.nn.functional as F
from MAMAMIA.nnUNet.nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from MAMAMIA.nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    
# eXpansion-EROsion network for binary segmentation refinement
class XERO(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=16, T=5, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.T = T  # Number of refinement steps
        self.numEpochs = 300
        self.numIterationsPerEpoch = 100

        self.xero = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, 3, kernel_size=3, padding=1),
            # probably include more convolutions
            nn.Softmax(dim=1)  # Output alpha_t ∈ [0, 1]
        )

        self.lr = 1e-3
        self.optimizer = torch.optim.AdamW(self.xero.parameters(), lr=self.lr, weight_decay=3e-5)
        self.loss = torch.nn.CrossEntropyLoss()
        self.grad_scaler = torch.GradScaler("cuda") if self.device.type == 'cuda' else None

        # need a custom dataloader since we want the entire image as input

        self.dataloader_tr = ...
        self.dataloader_vl = ...

        self.binarize = BinarySTE.apply

    def forward(self, logits: torch.Tensor):
        assert len(logits.shape) == 5, f"Expected logit dims to be (B, C, X, Y, Z), got {logits.shape}"
        pred_mask = torch.argmax(logits, dim=1, keepdim=True)[:, 1].float()     # shape: [B, X, Y, Z]
        print(f"pred_mask shape after argmax {pred_mask.shape}")
        assert torch.all(torch.unique(pred_mask) == torch.tensor([0., 1.]))
        # Iterative refinement
        xero = self.xero(logits)    # shape: [B, 3, X, Y, Z]
        print(f"xero shape {xero.shape}")
        # These are the tensors that we care about learning optimally
        exp = self.binarize(xero[:, 0])     # shape: [B, X, Y, Z]
        ero = self.binarize(xero[:, 1])     # shape: [B, X, Y, Z]
        print(f"exp shape {exp.shape}")
        print(f"ero shape {ero.shape}")
        for _ in range(self.T):
            masked_exp = diff_and(pred_mask, exp)
            masked_exp = diff_expand(masked_exp)
            pred_mask  = diff_or(pred_mask, masked_exp)

            inv_pred   = diff_not(pred_mask)
            
            masked_inv = diff_and(inv_pred, ero)
            masked_inv = diff_expand(masked_inv)
            inv_pred   = diff_or(inv_pred, masked_inv)

            pred_mask  = diff_not(inv_pred)
        return pred_mask
    
    def trainMe(self, predictor: nnUNetPredictor):
        segmentor = predictor.predict_logits_from_preprocessed_data
        for epoch in range(self.numEpochs):
            self.train()
            for batch in self.dataloader_tr:
                # read data directly from files so we can get the whole image at once

                data: torch.Tensor = batch['data']
                target: torch.Tensor = batch['target']
                data = data.to(self.device, non_blocking=True)
                if isinstance(target, list):
                    print("target is a list!")
                    target = [i.to(self.device, non_blocking=True) for i in target]
                else:
                    print("target is not a list!")
                    target = target.to(self.device, non_blocking=True)

                logits = segmentor(data)
                print(f"Logits shape: {logits.shape}")
                self.optimizer.zero_grad(set_to_none=True)
                # Autocast can be annoying
                # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
                # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
                # So autocast will only be active if we have a cuda device.
                with torch.autocast(self.device.type, enabled=True):
                    output = self(logits)
                    print(f"Shape output {output.shape}")
                    print(f"Shape target {target.shape}")
                    # del data
                    l = self.loss(output, target)
                if self.grad_scaler is not None:
                    self.grad_scaler.scale(l).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.xero.parameters(), 12)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    l.backward()
                    torch.nn.utils.clip_grad_norm_(self.xero.parameters(), 12)
                    self.optimizer.step()


class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return (x > 0.5).float()  # Round softmax output to binary

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# Assume kernel is binary, shape [B, X, Y, Z]
# Maybe make this learnable
def diff_expand(kernel: torch.Tensor):
    horizontal = F.max_pool3d(kernel, kernel_size=(2, 1), stride=1, padding=(1, 0))
    vertical   = F.max_pool3d(kernel, kernel_size=(1, 2), stride=1, padding=(0, 1))
    plus_expand = diff_or(horizontal, vertical)
    return plus_expand

# Element-wise OR
# Use De Morgan's law since plus (+) is not automorphic on [0, 1] in R (1 + 1 = 2)
# can't use clamp since gradient will get zeroed out when we see 1 + 1
def diff_or(mask: torch.Tensor, kernel: torch.Tensor):
    return diff_not(diff_and(diff_not(mask), diff_not(kernel)))

# Element-wise AND
# This is conveniently automorphic (any possible binary input will output valid binary)
def diff_and(mask: torch.Tensor, kernel: torch.Tensor):
    return mask * kernel

# Element-wise NOT
def diff_not(mask: torch.Tensor):
    return -(mask - 1)
