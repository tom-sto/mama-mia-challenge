import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingWithRestarts(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, cycle_steps, cycle_mult=1.0, maxLR=1e-3, minLR=1e-5, damping=1.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.cycle_mult = cycle_mult
        self.maxLR = maxLR
        self.minLR = minLR
        self.damping = damping
        
        self.cur_cycle = 0
        self.cycle_progress = 0
        self.next_cycle_step = warmup_steps + cycle_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step == 0:
            return [self.minLR for _ in self.base_lrs]

        # Warmup phase
        if step < self.warmup_steps:
            warmup_lr = self.minLR + (self.maxLR - self.minLR) * step / self.warmup_steps
            return [warmup_lr for _ in self.base_lrs]

        # Update cycle
        if step >= self.next_cycle_step:
            self.cur_cycle += 1
            self.cycle_steps = int(self.cycle_steps * self.cycle_mult)
            self.next_cycle_step = step + self.cycle_steps
            self.cycle_progress = 0
        else:
            self.cycle_progress = step - (self.next_cycle_step - self.cycle_steps)

        cycle_ratio = self.cycle_progress / self.cycle_steps
        damped_max_lr = self.maxLR * (self.damping ** self.cur_cycle)
        cosine_lr = self.minLR + 0.5 * (damped_max_lr - self.minLR) * (1 + math.cos(math.pi * cycle_ratio))

        return [cosine_lr for _ in self.base_lrs]