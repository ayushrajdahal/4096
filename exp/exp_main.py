import torch
import warnings
from exp.exp_basic import Exp_Basic
from typing import Literal
from dataclasses import dataclass
import torch.nn as nn

warnings.filterwarnings("ignore") # ignore warnings

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args=args)
    
    def _build_model(self):
        # remember: model_dict contains mappings from model names to their corresponding classes in exp/exp_basic.py
        model = self.model_dict[self.args.model].Model(self.args).float()

        # TODO: add multi-gpu support
        return model
    
    def _get_data(self, flag: Literal["train", "test", "val"]):
        data_set, data_loader = data_provider(self.args, flag) # TODO: implement data_provider
        return data_set, data_loader
    
    def _select_optimizer(self):
        optimizer = self.args.optimizer.lower()

        # maps optimizer names to its corresponding class
        
        map_optims = {
            'adam': torch.optim.Adam, # widely used and often performs well for various deep learning tasks, including time series forecasting
            'rmsprop': torch.optim.RMSprop, # particularly effective for recurrent neural networks like LSTMs
            'adagrad': torch.optim.Adagrad, # adapts the learning rate to the parameters, performing smaller updates for frequently occurring features and larger updates for infrequent features.
            # 'autocyclic': # TODO: implement AutoCyclic paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10410839
        }

        assert optimizer in map_optims.keys(), f"Optimizer {optimizer} not recognized. Available options are: {list(map_optims.keys())}"
        
        optimizer = map_optims[optimizer](self.model.parameters(), lr=self.args.learning_rate)
        
        return optimizer

    def _select_criterion(self, loss_type="mse"):
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == 'mae':
            return nn.L1Loss()
    
    def vali(self, vali_data, vali_loader, criterion):
        pass # TODO: need to implement dataloader to take care of this one