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

        if step < self.warmup_steps:
            scale = step / self.warmup_steps
            return [max(self.minLR, base_lr * scale) for base_lr in self.base_lrs]

        # 2. Cycle Management
        if step >= self.next_cycle_step:
            self.cur_cycle += 1
            self.cycle_progress = 0
            self.cycle_steps = int(self.cycle_steps * self.cycle_mult)
            self.next_cycle_step = step + self.cycle_steps
        else:
            self.cycle_progress = step - (self.next_cycle_step - self.cycle_steps)

        # 3. Cosine Annealing Phase
        cycle_ratio = self.cycle_progress / self.cycle_steps
        damped_factor = self.damping ** self.cur_cycle
        
        # Scale fluctuates between 0 and 1.0, multiplied by damping
        cosine_scale = 0.5 * (1 + math.cos(math.pi * cycle_ratio)) * damped_factor
        
        return [max(self.minLR, base_lr * cosine_scale) for base_lr in self.base_lrs]