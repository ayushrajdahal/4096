from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from typing import Literal
from utils.metrics import metric

warnings.filterwarnings("ignore") # ignore warnings

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args=args)
        self.last_epoch_stats = None
    
    def _build_model(self):
        # remember: model_dict contains mappings from model names to their corresponding classes in exp/exp_basic.py
        model = self.model_dict[self.args.model].Model(self.args).float()

        # TODO: add multi-gpu support
        return model
    
    def _get_data(self, flag: Literal["train", "test", "val"]):
        data_set, data_loader = data_provider(self.args, flag) # TODO: implement data_provider
        return data_set, data_loader
    
    def _select_optimizer(self):
        if self.args.optimizer:
            optimizer = self.args.optimizer.lower()
        else:
            optimizer = 'adam'

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
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
            
                # corresponding timestamps for the batches: either integer values or 

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        mse_criterion = self._select_criterion('mse')
        mae_criterion = self._select_criterion('mae')

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_mse = []
            train_mae = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        
                        mse_loss = mse_criterion(outputs, batch_y)
                        mae_loss = mae_criterion(outputs, batch_y)
                        
                        train_mse.append(mse_loss.item())
                        train_mae.append(mae_loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    mse_loss = mse_criterion(outputs, batch_y)
                    mae_loss = mae_criterion(outputs, batch_y)
                    train_mse.append(mse_loss.item())
                    train_mae.append(mae_loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | MSE: {2:.7f} | MAE: {3:.7f}".format(i + 1, epoch + 1, mse_loss.item(), mae_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(mse_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    mse_loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            train_mse = np.average(train_mse)
            train_mae = np.average(train_mae)
            
            vali_mse = self.vali(vali_data, vali_loader, mse_criterion)
            vali_mae = self.vali(vali_data, vali_loader, mae_criterion)
            
            test_mse = self.vali(test_data, test_loader, mse_criterion)
            test_mae = self.vali(test_data, test_loader, mae_criterion)

            print(f"Epoch: {epoch+1}, Steps: {train_steps} | Train MSE: {train_mse:.7f} Vali MSE: {vali_mse:.7f} Test MSE: {test_mse:.7f} | Train MAE: {train_mae:.7f} | Vali MAE: {vali_mae:.7f} | Test MAE: {test_mae:.7f}")
            early_stopping(vali_mse, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.last_epoch_stats = {
            'train_mse': train_mse,
            'vali_mse': vali_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'vali_mae': vali_mae,
            'test_mae': test_mae
        }

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # # dtw calculation
        # if self.args.use_dtw:
        #     dtw_list = []
        #     manhattan_distance = lambda x, y: np.abs(x - y)
        #     for i in range(preds.shape[0]):
        #         x = preds[i].reshape(-1,1)
        #         y = trues[i].reshape(-1,1)
        #         if i % 100 == 0:
        #             print("calculating dtw iter:", i)
        #         d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
        #         dtw_list.append(d)
        #     dtw = np.array(dtw_list).mean()
        # else:
        #     dtw = 'not calculated'
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}, rmse:{rmse}, mape:{mape}, mspe:{mspe}')
        f = open("result_long_term_forecast.txt", 'a')
        f.write(f"Model: {self.args.model} | Dataset: {self.args.data_path.split('.csv')[0]} | Seq Len: {self.args.seq_len} | Label Len: {self.args.label_len} | Pred Len: {self.args.pred_len} | Enc In: {self.args.enc_in} | Dec In: {self.args.dec_in} | Enc Layers: {self.args.e_layers} | Dec Layers: {self.args.d_layers}" + "  \n")
        f.write(f'mse:{mse}, mae:{mae}, rmse:{rmse}, mape:{mape}, mspe:{mspe}\n')
        f.write(f"Last Epoch Stats: {self.last_epoch_stats}\n")
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return