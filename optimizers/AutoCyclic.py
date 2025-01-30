"""
AutoCyclic: Deep Learning Optimizer for Time Series Data Prediction

Links:
- https://ieeexplore.ieee.org/document/10410839
- https://github.com/wtfish/AutoCyclic/blob/main/autoCyclic.ipynb
"""


import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_forecasting

class AutoCyclicLR(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, step_size, last_epoch=-1):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.data = None
        super(AutoCyclicLR, self).__init__(optimizer, last_epoch) 
    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size))
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)
        lr = [self.base_lr + (self.max_lr - self.base_lr) * (1 + math.cos(math.pi * x)) / 2 * (1 + self.get_batch_variance()) for _ in self.base_lrs]
        return lr
    def get_batch_variance(self):
        # AutoCorrelation+sigmoid
        if self.data is None:
            return 1
        step_var=[]
        for items in self.data:
            SGLayer = nn.Sigmoid()
            output=pytorch_forecasting.autocorrelation(items)
            output=torch.nan_to_num(output, nan=0)
            output=SGLayer(output)
            step_var.append(torch.var(output))
        tensor_batch_step_var = torch.tensor(step_var)
        mean_step=torch.mean(tensor_batch_step_var)
        batch_variance = mean_step.numpy()
        return batch_variance    
    def set_batch_data(self, data):
        self.data = data