from data_provider.data_loader import Dataset_Energy_hour, Dataset_ETT_hour
from torch.utils import DataLoader

data_dict = {
    'energy': Dataset_Energy_hour,
    'ETTh1': Dataset_ETT_hour,
}

def data_provider(args, flag:str):
    Data = data_dict[args.data]
    
    timeenc = 0 if args.embed != 'timeF' else 1
    
    # NOTE:
    # timeF stands for "time features". It is used to determine the type of embedding for temporal data. 
    # When args.embed is set to timeF, it indicates that time features should be used for embedding, and timeenc is set to 1. 
    # This is used in the TemporalEmbedding and TimeFeatureEmbedding classes in Embed.py.

    shuffle_flag = False if flag.lower() == 'test' else True
    drop_last = False               # whether to drop the last incomplete batch
    batch_size = args.batch_size    # batch size of the dataloader
    freq = args.freq                # frequency of the data --  TODO: what does it mean?

    # TODO: add logic for anomaly detection, classification, etc.

    if args.data == 'm4': # m4 benchmark dataset: https://paperswithcode.com/dataset/m4
        drop_last = False
    
    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )

    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader