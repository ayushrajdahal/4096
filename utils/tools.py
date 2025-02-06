import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjust the learning rate based on the epoch and args
    """
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * 0.5 ** ((epoch-1) // 1)} # halves iteself in every epoch
    elif args.lradj == "type2": # epoch numbers mapped to learning rates
        lr_adjust = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))} # NOTE: half a cosine goes from 1 to -1 smoothly => lr_adjust goes form 1 to 0 as it approaches args.training_epochs
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updated learning rate to {lr}')

class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0):
        self.patience = patience    # how many times to let val loss go above min
        self.verbose = verbose      # prints out validation loss decreases whenever new min occurs and model checkpoint is saved
        self.delta = delta          # alter the value of lowest val loss when comparing
        self.counter = 0            # count no. of times val loss has gone above min
        self.early_stop = False     # set to true when we need to stop training
        self.val_loss_min = np.inf  # minimum validation loss

    def __call__(self, val_loss, model, path):
        """
        Check if early stopping should be triggered.
        """
        if self.val_loss_min is np.inf:
            self.val_loss_min = val_loss
        elif val_loss >= self.val_loss_min - self.delta:
            self.counter += 1
            print(f"EarlyStopping patience counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model, path)
            self.val_loss_min = val_loss
            self.counter = 0 # reset the counter
        

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min} => {val_loss})')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')


class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    
    Example usage:
    d = dotdict()
    d.key = 'value'  # Sets d['key'] to 'value'
    print(d.key)     # Gets d['key'], prints 'value'
    del d.key        # Deletes d['key']

    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        # standardize the data
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        # revert the standardization
        return data * self.std + self.mean

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend() 
    plt.savefig(name, bbox_inches='tight') # bbox_inches: Bounding box in inches -- only the given portion of the figure is saved. If 'tight', try to figure out the tight bbox of the figure.

# MISSING: adjustment() for anomaly detection

def cal_accuracy(y_pred, y_true):
    # calculate the accuracy
    return np.mean(y_pred == y_true)
