import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SeriesDecomp(nn.Module):
    """Decompose the series into trend and seasonal components."""
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        # x: [Batch, Length, Channel]
        trend = self.avg(x.transpose(1,2)).transpose(1,2)
        seasonal = x - trend
        return seasonal, trend

class AutoCorrelation(nn.Module):
    """AutoCorrelation layer for capturing periodic patterns."""
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False, n_heads=8):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.n_heads = n_heads
    
    def time_delay_agg_training(self, values, corr):
        """
        Time-delay aggregation training process
        Args:
            values: [Batch, Head, Length, Channel]
            corr: [Batch, Head, Length, Length]
        Returns:
            [Batch, Head, Length, Channel]
        """
        batch, head, channel, length = values.shape
        
        # Find top k correlation positions for each query
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(corr, dim=1, keepdim=True)
        weights = torch.softmax(corr - mean_value, dim=-1)
        
        # Ensure weights and values have compatible dimensions for matmul
        weights = weights.transpose(-1, -2)  # [B, H, Length, Length]
        V = torch.matmul(weights, values.transpose(-1, -2))  # [B, H, Length, Channel]
        
        return V.transpose(-1, -2)  # Return to original shape

    def forward(self, queries, keys, values):
        B, L, E = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # Split heads
        queries = queries.reshape(B, L, H, -1)
        keys = keys.reshape(B, S, H, -1)
        values = values.reshape(B, S, H, -1)
        
        # Transpose for attention calculation
        queries = queries.permute(0, 2, 1, 3)  # [B, H, L, E/H]
        keys = keys.permute(0, 2, 1, 3)  # [B, H, S, E/H]
        values = values.permute(0, 2, 3, 1)  # [B, H, E/H, S]
        
        # Calculate correlations using FFT
        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)

        # Time delay aggregation
        V = self.time_delay_agg_training(values, corr)
        
        # Reshape output
        out = V.permute(0, 2, 1, 3).reshape(B, L, -1)
        
        return out if not self.output_attention else (out, None)
class Model(nn.Module):
    """
    Autoformer implementation with series decomposition and auto-correlation
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomposition
        kernel_size = configs.moving_avg if hasattr(configs, 'moving_avg') else 25
        self.decomp = SeriesDecomp(kernel_size)

        # Embedding
        self.enc_embedding = nn.Linear(configs.enc_in, configs.d_model)
        self.dec_embedding = nn.Linear(configs.dec_in, configs.d_model)

        # Encoder
        self.encoder = nn.ModuleList([
            AutoCorrelation(
                mask_flag=False, 
                factor=configs.factor,
                attention_dropout=configs.dropout,
                output_attention=configs.output_attention,
                n_heads=configs.n_heads
            ) for _ in range(configs.e_layers)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            AutoCorrelation(
                mask_flag=True,
                factor=configs.factor,
                attention_dropout=configs.dropout,
                output_attention=configs.output_attention
            ) for _ in range(configs.d_layers)
        ])

        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        enc_out = self.enc_embedding(x_enc)
        dec_out = self.dec_embedding(x_dec)
        
        # Encoder
        enc_seasonal_list = []
        enc_trend_list = []
        
        enc_seasonal, enc_trend = self.decomp(enc_out)
        for encoder in self.encoder:
            enc_seasonal = encoder(enc_seasonal, enc_seasonal, enc_seasonal)
            enc_seasonal_list.append(enc_seasonal)
            enc_trend_list.append(enc_trend)
        
        # Decoder
        seasonal_part, trend_part = self.decomp(dec_out)
        
        for decoder in self.decoder:
            seasonal_part = decoder(seasonal_part, enc_seasonal, enc_seasonal)
            trend_part = trend_part + enc_trend_list[-1]
        
        dec_out = seasonal_part + trend_part
        
        # Final projection
        dec_out = self.projection(dec_out)
        
        if self.output_attention:
            return dec_out, None
        else:
            return dec_out  # [B, L, D]