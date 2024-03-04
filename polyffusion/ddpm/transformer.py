import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ddpm.unet import TimeEmbedding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Positional Encodingを保持するテーブルを作成
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # 勾配計算を無効にして、値が更新されないようにする
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Positional Encodingを入力xに加算
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, nhead, max_len=512):
        super(TransformerEncoderModel, self).__init__()
        self.positional_encoding = PositionalEncoding(input_dim, max_len)
        self.time_embedding = TimeEmbedding(input_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True),
            num_layers=num_layers,
        )

    def forward(self, x, mask, t):
        """forward

        Args:
            x (torch.Tensor): shape=(batch, 1, sequence_length, dim)
            mask (torch.Tensor): shape=(batch, sequence_length)
            t (torch.Tensor): shape=(batch)

        Returns:
            torch.Tensor: output tensor, shape=(batch, 1, sequence_length, dim)
        """
        x = x.squeeze(1)
        src = self.positional_encoding(x)
        time_embedding = self.time_embedding(t).unsqueeze(1)
        src = src + time_embedding
        output = self.transformer_encoder(src, src_key_padding_mask=~mask.bool())

        # パディングされた部分を入力の値で置き換え
        output = torch.where(mask.unsqueeze(-1).bool(), output, x)

        return output.unsqueeze(1)
