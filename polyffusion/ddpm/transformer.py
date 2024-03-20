import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ddpm.unet import TimeEmbedding

def get_sample_lengths_from_src_key_padding_mask(src_key_padding_mask):
    """
    PyTorchのsrc_key_padding_maskからバッチ内のサンプルの長さのリストを取得する関数

    Args:
        src_key_padding_mask (torch.Tensor): shape=(batch_size, src_len) PAD = 1, NOT PAD = 0

    Returns:
        sample_lengths (list): list of each sample length
    """
    batch_size, src_len = src_key_padding_mask.size()
    sample_lengths = []

    for batch_idx in range(batch_size):
        sample_mask = src_key_padding_mask[batch_idx]
        sample_length = (sample_mask == 0).sum().item()
        sample_lengths.append(sample_length)

    return sample_lengths

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, n_dim_divide=1):
        super(PositionalEncoding, self).__init__()
        # Positional Encodingを保持するテーブルを作成
        if d_model % n_dim_divide != 0:
            raise ValueError("dim_divide must be a divisor of d_model")
        self.n_dim_divide = n_dim_divide
        pe_dim = d_model // n_dim_divide
        pe = torch.zeros(max_len, pe_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pe_dim, 2).float() * (-math.log(10000.0) / pe_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # 勾配計算を無効にして、値が更新されないようにする
        self.register_buffer('pe', pe)

    def reverse_pe_add(self, x, mask):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            mask: Tensor, shape [batch_size, seq_len]
        """
        lengths = get_sample_lengths_from_src_key_padding_mask(mask)
        for i, length in enumerate(lengths):
            x[i, :length, :] = x[i, :length, :] + self.pe[:, :length, :].flip(dims=[1, 2]).repeat(1, 1, self.n_dim_divide)
        return x

    def forward(self, x, mask):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Positional Encodingを入力xに加算
        x = self.reverse_pe_add(x, mask) + self.pe[:, :x.size(1), :].repeat(1, 1, self.n_dim_divide)
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
            mask (torch.Tensor): shape=(batch, sequence_length), mask for PAD token or value
            t (torch.Tensor): shape=(batch)

        Returns:
            torch.Tensor: output tensor, shape=(batch, 1, sequence_length, dim)
        """
        x = x.squeeze(1)
        src = self.positional_encoding(x, mask)
        time_embedding = self.time_embedding(t).unsqueeze(1)
        src = src + time_embedding
        output = self.transformer_encoder(src, src_key_padding_mask=~mask.bool())

        # パディングされた部分を入力の値で置き換え
        output = torch.where(mask.unsqueeze(-1).bool(), output, x)

        return output.unsqueeze(1)


if __name__ == "__main__":
    model = TransformerEncoderModel(512, 2048, 6, 8, 128)
    x = torch.randn(4, 1, 128, 512)
    mask = torch.zeros(4, 128)
    mask[0, 10:] = 1
    mask[0, 30:] = 1
    mask[0, 80:] = 1

    output = model(x, mask, torch.tensor([0, 1, 2, 3]))

    print(output.size())
