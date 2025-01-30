import os
import pandas as pd
from data_provider.data_loader import Dataset_Energy_hour

# Define a class to simulate the arguments
class Args:
    def __init__(self):
        self.seq_len = 24 * 4 * 4
        self.label_len = 24 * 4
        self.pred_len = 24 * 4
        self.features = 'M'
        self.target = 'nat_demand'
        self.scale = True
        self.timeenc = 0
        self.freq = 'h'
        self.scaler_name = 'standard'
        self.root_path = 'datasets'
        self.data_path = 'load_forecasting.csv'

# Create instances of Dataset_Energy_hour for train, test, and val
args = Args()

for flag in ['train', 'test', 'val']:
    dataset = Dataset_Energy_hour(root_path=args.root_path, flag=flag, size=[args.seq_len, args.label_len, args.pred_len],
                                  features=args.features, data_path=args.data_path, target=args.target, scale=args.scale,
                                  timeenc=args.timeenc, freq=args.freq, scaler_name=args.scaler_name)
    print(f"Flag: {flag}")
    print(f"border1: {dataset.border1}")
    print(f"border2: {dataset.border2}")
    print(f"Data slice from border1 to border2:\n{dataset.data[dataset.border1:dataset.border2]}")
    print()