"""
train latent sequence transformer
usage example(using pretrained DisentangleVAE):
python train_transformer.py --output_dir=result/debug --batch_size=4 --max_epoch=4 --pin_memory=False --n_layers=2 --custom_vae_class=polyffusion.polydis.model.DisentangleVAE --custom_vae_yaml=result/vae_0.1_0.0001/02-01_174905/params.yaml --custom_vae_ckpt=result/vae_0.1_0.0001/02-01_174905/chkpts/best_weights.pt --is_sample
"""
from __future__ import annotations
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "polyffusion"))
sys.path.append(os.path.join(os.path.dirname(__file__), "polyffusion", "chord_extractor"))
from typing import Callable
import torch
from torch import nn
from torch.nn import MultiheadAttention, LayerNorm
from torch.nn.functional import relu
import numpy as np
import yaml
import importlib
from omegaconf import OmegaConf

import math
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.to(device)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).permute(1, 0, 2)

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, device=device)
        self.linear2 = nn.Linear(d_ff, d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(relu(self.linear1(x)))

class TransformerDecoderLayerWithoutSrcTgtAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
        layer_norm_eps: float,
        device: torch.device = torch.device("cpu"),
    ):
        """Transformer decoder without source-target cross attention

        Args:
            d_model (int): model dim
            d_ff (int): hidden dim in feed-forward(ff network has 2 Linear layers)
            heads_num (int): number of head
            dropout_rate (float): dropout rate
            layer_norm_eps (float): epsilon parameter in LayerNorm
            device (torch.device): device
        """
        super().__init__()
        self.self_attention = MultiheadAttention(d_model, heads_num, batch_first=True, device=device)
        self.dropout_self_attention = nn.Dropout(dropout_rate)
        self.layer_norm_self_attention = LayerNorm(d_model, eps=layer_norm_eps, device=device)

        self.ffn = FFN(d_model, d_ff, device=device)
        self.dropout_ffn = nn.Dropout(dropout_rate)
        self.layer_norm_ffn = LayerNorm(d_model, eps=layer_norm_eps, device=device)

    def forward(
        self,
        tgt: torch.Tensor,  # Decoder input
        mask_self: torch.Tensor,
    ) -> torch.Tensor:
        """decoder layer forward

        Args:
            tgt (torch.Tensor): (batch, time, dim)
            mask_self (torch.Tensor): (batch, time, time)

        Returns:
            torch.Tensor: (batch, time, dim)
        """
        x = self.layer_norm_self_attention(
            tgt + self.__self_attention_block(tgt, mask_self)
        )

        x = self.layer_norm_ffn(x + self.__feed_forward_block(x))

        return x

    def __self_attention_block(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        x, attention_weights = self.self_attention(x, x, x, attn_mask=mask)
        return self.dropout_self_attention(x)

    def __feed_forward_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout_ffn(self.ffn(x))


class TransformerDecoderWithoutSrcTgtAttention(nn.Module):
    def __init__(
        self,
        max_len: int,
        d_model: int,
        n_layers: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
        layer_norm_eps: float,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, dropout_rate, max_len, device)
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayerWithoutSrcTgtAttention(
                    d_model, d_ff, heads_num, dropout_rate, layer_norm_eps, device
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,  # Decoder input
        mask_self: torch.Tensor,
    ) -> torch.Tensor:
        """decoder layer forward

        Args:
            tgt (torch.Tensor): (batch, time, dim)
            mask_self (torch.Tensor): (batch, time, time)

        Returns:
            torch.Tensor: (batch, time, dim)
        """
        # tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(
                tgt,
                mask_self,
            )
        return tgt

class GenerativeTransformerModel(nn.Module):
    def __init__(
        self,
        max_len: int = 256,  # max_len = 小節数とすると，
        d_model: int = 512,
        heads_num: int = 8,
        d_ff: int = 2048,
        n_layers: int = 6,
        dropout_rate: float = 0.1,
        layer_norm_eps: float = 1e-5,
        device: torch.device = torch.device("cpu"),
    ):

        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.heads_num = heads_num
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.device = device

        self.decoder = TransformerDecoderWithoutSrcTgtAttention(
            max_len,
            d_model,
            n_layers,
            d_ff,
            heads_num,
            dropout_rate,
            layer_norm_eps,
            device,
        )

        self.sequence_linear = nn.Linear(d_model, d_model, device=device)
        self.stop_linear = nn.Linear(d_model, 1, device=device)


    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        ----------
        tgt : torch.Tensor
            ベクトル系列. [batch_size, max_len, d_model]
        """

        # target系列の"0~max_len-1"(max_len-1系列)までを入力し、"1~max_len"(max_len-1系列)を予測する
        mask_self_attn = self._subsequent_mask(tgt)
        dec_output = self.decoder(tgt, mask_self_attn)
        dec_output_seq = self.sequence_linear(dec_output)
        dec_output_stop = torch.sigmoid(self.stop_linear(dec_output))

        return dec_output_seq, dec_output_stop

    def _subsequent_mask(self, x: torch.Tensor) -> torch.Tensor:
        """DecoderのMasked-Attentionに使用するmaskを作成する.
        Parameters:
        ----------
        x : torch.Tensor
            単語のトークン列. [batch_size, max_len, d_model]
        """
        batch_size = x.size(0)
        max_len = x.size(1)
        return (
            torch.tril(torch.ones(batch_size * self.heads_num, max_len, max_len)).eq(0).to(self.device)
        )


from polyffusion.utils import load_pretrained_chd_enc_dec, load_pretrained_txt_enc
from polyffusion.dirs import PT_CHD_8BAR_PATH, PT_POLYDIS_PATH, POP909_DATA_DIR

# データローダーの読み込み
from typing import Any
import os
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import numpy as np
from polyffusion.dirs import TRAIN_SPLIT_DIR
from polyffusion.data.dataset import PianoOrchDataset, DataSampleNpz
from polyffusion.data.datasample import DataSample
from polyffusion.dl_modules.chord_enc import RnnEncoder as ChordEncoder
from polyffusion.dl_modules.txt_enc import TextureEncoder
from polyffusion.utils import (
    pr_mat_pitch_shift, prmat2c_to_midi_file, chd_to_onehot, chd_pitch_shift,
    chd_to_midi_file, estx_to_midi_file, pianotree_pitch_shift, prmat_to_midi_file,
    read_dict, onehot_to_chd
)


class WholeSongDataset(Dataset):
    def __init__(self, dataset: list[DataSampleNpz]) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DataSampleNpz:
        return self.dataset[idx]

    @classmethod
    def load_with_song_paths(cls, song_paths, **kwargs):
        data_samples = [DataSampleNpz(song_path, **kwargs) for song_path in song_paths]
        return cls(data_samples)

    @classmethod
    def remove_not_4beat_songs(cls, song_list: list[str]) -> list[str]:
        dataset = cls.load_with_song_paths(song_list)
        new_song_list = []
        for song in dataset:
            if len(song.get_whole_song_data()[0]) != 0:
                new_song_list.append(song.song_fn)
        return new_song_list

    @classmethod
    def load_train_and_valid_sets(cls, **kwargs)\
            -> tuple[WholeSongDataset, WholeSongDataset]:
        split = read_dict(os.path.join(TRAIN_SPLIT_DIR, "pop909.pickle"))
        # exclude songs not being 4 beat
        train_list = cls.remove_not_4beat_songs(split[0])
        val_list = cls.remove_not_4beat_songs(split[1])
        print("load train valid set with:", kwargs)
        print(f"number of data: train: {len(train_list)}, val: {len(val_list)}")
        return cls.load_with_song_paths(train_list), cls.load_with_song_paths(
                                            val_list
                                        )


from vae import add_filename_suffix, replace_extension, get_data_for_single_midi  # TODO: utilsとかにまとめる
class NBarsDataSample(Dataset):
    # TODO: これを改造する
    def __init__(self, data_samples: list[DataSample]) -> None:
        super().__init__()
        self.data_samples = data_samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        return self.data_samples[index]

    @classmethod
    def load_with_song_paths(
        cls, song_paths, data_dir=POP909_DATA_DIR, debug=False, **kwargs,
    ):
        data_samples = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for song_path in tqdm(song_paths, desc="DataSample loading"):
                mid_song_path = os.path.join(
                    data_dir,
                    song_path,
                )
                data_samples += [
                    DataSample(
                        get_data_for_single_midi(
                            mid_song_path,
                            os.path.join(temp_dir, "chords_extracted.out")
                        ),
                        kwargs["n_bars"]
                    )
                ]
        return cls(data_samples)

    @classmethod
    def load_train_and_valid_sets(cls, debug=False, **kwargs,):
        if debug:
            split = read_dict(os.path.join(TRAIN_SPLIT_DIR, "pop909_debug32.pickle"))
        else:
            split = read_dict(os.path.join(TRAIN_SPLIT_DIR, "pop909.pickle"))
        split = list(split)
        split[0] = list(map(lambda x: add_filename_suffix(x, "_flatten"), split[0]))
        split[1] = list(map(lambda x: add_filename_suffix(x, "_flatten"), split[1]))
        split[0] = list(map(lambda x: replace_extension(x, ".mid"), split[0]))
        split[1] = list(map(lambda x: replace_extension(x, ".mid"), split[1]))

        print("load train valid set with:", kwargs)
        return cls.load_with_song_paths(
            split[0], debug=debug, **kwargs
        ), cls.load_with_song_paths(split[1], debug=debug, **kwargs)

    @classmethod
    def load_valid_sets(cls, debug=False, **kwargs,):
        if debug:
            split = read_dict(os.path.join(TRAIN_SPLIT_DIR, "pop909_debug32.pickle"))
        else:
            split = read_dict(os.path.join(TRAIN_SPLIT_DIR, "pop909.pickle"))
        split = list(split)
        split[1] = list(map(lambda x: add_filename_suffix(x, "_flatten"), split[1]))
        split[1] = list(map(lambda x: replace_extension(x, ".mid"), split[1]))
        print("load valid set with:", kwargs)
        return cls.load_with_song_paths(split[1], debug=debug, **kwargs)

    @classmethod
    def load_set_with_mid_dir_path(cls, mid_dir_path, **kwargs):
        directory = Path(mid_dir_path)
        midi_file_names = [f.name for f in directory.iterdir() if f.is_file() and f.suffix == ".mid"]
        print(f"loaded midi files:", midi_file_names)
        return cls.load_with_song_paths(midi_file_names, mid_dir_path, **kwargs)


def sample_shift():
    return np.random.choice(np.arange(-6, 6), 1)[0]

class DefaultVAECollator:
    def __init__(self, chord_enc: ChordEncoder, txt_enc: TextureEncoder):
        self.chord_enc = chord_enc
        self.txt_enc = txt_enc

    def _encode_chord(self, chord):
        if self.chord_enc is not None:
            # z_list = []
            # for chord_seg in chord.split(8, 1):  # (#B, 8, 36) * 4
            #     z_seg = self.chord_enc(chord_seg).mean
            #     z_list.append(z_seg)
            # z = torch.stack(z_list, dim=1)
            z = self.chord_enc(chord).mean
            z = z.unsqueeze(1)  # (#B, 1, 512)
            return z
        else:
            chord_flatten = torch.reshape(
                chord, (-1, 1, chord.shape[1] * chord.shape[2])
            )
            return chord_flatten

    def _encode_txt(self, prmat):
        z_list = []
        if self.txt_enc is not None:
            for prmat_seg in prmat.split(32, 1):  # (#B, 32, 128) * 4
                z_seg = self.txt_enc(prmat_seg).mean
                z_list.append(z_seg)
            z = torch.cat(z_list, dim=-1)
            z = z.unsqueeze(1)  # (#B, 1, 256*4)
            return z
        else:
            # print(f"unencoded txt: {prmat.shape}")
            return prmat

    def __call__(
            self,
            batch: list[DataSampleNpz],
            shift: bool) -> tuple[
                torch.Tensor,
                list[torch.Tensor],
                list[torch.Tensor],
                list[torch.Tensor],
                list[torch.Tensor]]:

        prmat2c = []
        pnotree = []
        chord = []
        prmat = []
        song_fn = []

        cond_chords = []
        cond_txts = []

        max_sequence_length = 0

        seq_length = []

        for b in batch:
            # b[0]: seg_pnotree; b[1]: seg_pnotree_y
            song_fn = b.song_fn
            b = b.get_whole_song_data()
            seg_prmat2cs = b[0]
            seg_pnotrees = b[1]
            seg_chords = b[2]
            seg_prmats = b[3]

            # update max_sequence_length
            if len(seg_prmat2cs) > max_sequence_length:
                max_sequence_length = len(seg_prmat2cs)

            seq_length.append(len(seg_prmat2cs))

            prmat2cs = []
            pnotrees = []
            chords = []
            prmats = []

            if shift:
                shift_pitch = sample_shift()
            else:
                shift_pitch = 0

            # loop for each segment
            for i in range(len(seg_prmat2cs)):
                seg_prmat2c = pr_mat_pitch_shift(seg_prmat2cs[i].detach().cpu().numpy(), shift_pitch)
                seg_pnotree = pianotree_pitch_shift(seg_pnotrees[i].detach().cpu().numpy(), shift_pitch)
                seg_chord = chd_pitch_shift(onehot_to_chd(seg_chords[i].detach().cpu().numpy()).astype(int), shift_pitch)
                seg_prmat = pr_mat_pitch_shift(seg_prmats[i].detach().cpu().numpy(), shift_pitch)

                seg_chord = chd_to_onehot(seg_chord)

                prmat2cs.append(seg_prmat2c)
                pnotrees.append(seg_pnotree)
                chords.append(seg_chord)
                prmats.append(seg_prmat)

            # encode one song
            prmat2cs = torch.Tensor(np.array(prmat2cs)).float()
            pnotrees = torch.Tensor(np.array(pnotrees)).long()
            chords = torch.Tensor(np.array(chords)).float()
            prmats = torch.Tensor(np.array(prmats)).float()

            if self.chord_enc is not None:
                cond_chords.append(self._encode_chord(chords))
            if self.txt_enc is not None:
                cond_txts.append(self._encode_txt(prmats))


            prmat2c.append(prmat2cs)
            pnotree.append(pnotrees)
            chord.append(chords)
            prmat.append(prmats)

            if len(b) > 4:
                song_fn.append(b[4])

        # pad to max_sequence_length
        if self.chord_enc is not None:
            cond_chords = torch.nn.utils.rnn.pad_sequence(cond_chords, batch_first=True)
        if self.txt_enc is not None:
            cond_txts = torch.nn.utils.rnn.pad_sequence(cond_txts, batch_first=True)

        cond_chords = torch.Tensor(cond_chords)
        cond_txts = torch.Tensor(cond_txts)

        cond = torch.cat([cond_chords, cond_txts], dim=-1).squeeze(2)

        if len(song_fn) > 0:
            return cond, seq_length, prmat2c, pnotree, chord, prmat, song_fn
        else:
            return cond, seq_length, prmat2c, pnotree, chord, prmat


class CustomVAECollator:
    def __init__(self, vae: nn.Module, is_sample: bool=True):
        """_summary_

        Args:
            vae (nn.Module): vae
            is_sample (bool, optional): sampled latent value or estimated mean of distribution. Defaults to True.
        """
        self.vae = vae
        self.is_sample = is_sample
        self.timesteps_per_sample = 32

    def _encode(self, prmat, chd):
        """extract latent representation from prmat and chd

        Args:
            prmat (torch.Tensor, shape=(n_bars, 16, 128)): pianoroll matrix
            chd (torch.Tensor, shape=(n_bars, 4, 36): chord matrix

        Returns:
            torch.Tensor: latent representation. shape=()
        """
        # for prmat_seg in range(len(prmat)):
        dist_chd, dist_rhy = self.vae.inference_encode(prmat, chd)
        if self.is_sample:
            z_chd = dist_chd.rsample()
            z_rhy = dist_rhy.rsample()
        else:
            z_chd = dist_chd.mean
            z_rhy = dist_rhy.mean
        z = torch.cat((z_chd, z_rhy), dim=1)
        z = z.unsqueeze(1)  # (#B, 1, 256*4)
        return z

    def __call__(
            self,
            batch: list[DataSampleNpz],
            shift: bool) -> tuple[
                torch.Tensor,
                list[torch.Tensor],
                list[torch.Tensor],
                list[torch.Tensor],
                list[torch.Tensor]]:

        prmat2c = []
        pnotree = []
        chord = []
        prmat = []
        song_fn = []

        conditions = []

        max_sequence_length = 0

        seq_length = []

        for b in batch:
            # b[0]: seg_pnotree; b[1]: seg_pnotree_y
            b = b.get_whole_song_data()
            seg_prmat2cs = b[0]
            seg_pnotrees = b[1]
            seg_chords = b[2]
            seg_prmats = b[3]

            # update max_sequence_length
            if len(seg_prmat2cs) > max_sequence_length:
                max_sequence_length = len(seg_prmat2cs)

            seq_length.append(len(seg_prmat2cs))

            prmat2cs = []
            pnotrees = []
            chords = []
            prmats = []

            if shift:
                shift_pitch = sample_shift()
            else:
                shift_pitch = 0

            # loop for each segment
            for i in range(len(seg_prmat2cs)):
                seg_prmat2c = pr_mat_pitch_shift(seg_prmat2cs[i].detach().cpu().numpy(), shift_pitch)
                seg_pnotree = pianotree_pitch_shift(seg_pnotrees[i].detach().cpu().numpy(), shift_pitch)
                seg_chord = chd_pitch_shift(onehot_to_chd(seg_chords[i].detach().cpu().numpy()).astype(int), shift_pitch)
                seg_prmat = pr_mat_pitch_shift(seg_prmats[i].detach().cpu().numpy(), shift_pitch)

                seg_chord = chd_to_onehot(seg_chord)

                prmat2cs.append(seg_prmat2c)
                pnotrees.append(seg_pnotree)
                chords.append(seg_chord)
                prmats.append(seg_prmat)

            # encode one song
            prmat2cs = torch.Tensor(np.array(prmat2cs)).float()
            pnotrees = torch.Tensor(np.array(pnotrees)).long()
            chords = torch.Tensor(np.array(chords)).float()
            prmats = torch.Tensor(np.array(prmats)).float()

            conditions.append(self._encode(prmats, chords))

            prmat2c.append(prmat2cs)
            pnotree.append(pnotrees)
            chord.append(chords)
            prmat.append(prmats)

            if len(b) > 4:
                song_fn.append(b[4])

        # pad to max_sequence_length
        conditions = torch.nn.utils.rnn.pad_sequence(conditions, batch_first=True)
        conditions = conditions.squeeze(2)

        if len(song_fn) > 0:
            return conditions, seq_length, prmat2c, pnotree, chord, prmat, song_fn
        else:
            return conditions, seq_length, prmat2c, pnotree, chord, prmat


class CollatorWrapper:
    def __init__(self, collator: Callable, shift: bool):
        self.collator = collator
        self.shift = shift

    def __call__(self, batch: list[DataSampleNpz]) -> Any:
        return self.collator(batch, self.shift)


def get_train_val_dataloaders_8bars(
    batch_size, num_workers=0, pin_memory=False, debug=False, collate_fn:Callable=callable, **kwargs
):
    train_dataset, val_dataset = WholeSongDataset.load_train_and_valid_sets(
        **kwargs
    )
    train_collate_fn = CollatorWrapper(collate_fn, shift=True)
    valid_collate_fn = CollatorWrapper(collate_fn, shift=False)
    train_dl = DataLoader(
        train_dataset,
        batch_size,
        True,
        collate_fn=train_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size,
        True,
        collate_fn=valid_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    print(
        f"Dataloader ready: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}, {kwargs}"
    )
    return train_dl, val_dl


def get_train_val_dataloaders_n_bars(
    batch_size, num_workers=0, pin_memory=False, debug=False, n_bars=1, collate_fn:Callable=callable, **kwargs
):
    train_dataset, val_dataset = NBarsDataSample.load_train_and_valid_sets(
        debug=debug,
        n_bars=n_bars,
        **kwargs
    )
    train_collate_fn = CollatorWrapper(collate_fn, shift=True)
    valid_collate_fn = CollatorWrapper(collate_fn, shift=False)
    train_dl = DataLoader(
        train_dataset,
        batch_size,
        True,
        collate_fn=train_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size,
        True,
        collate_fn=valid_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    print(
        f"Dataloader ready: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}, {kwargs}"
    )
    return train_dl, val_dl


import os
import argparse
from datetime import datetime
from tqdm import tqdm
import json
from torch.utils.tensorboard.writer import SummaryWriter


def adjust_learning_rate(optimizer, step_num, warmup_step=4000, learning_rate=0.001):
    lr = learning_rate * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="result/transformer")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=10)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--heads_num", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--weight_stop_estimation", type=float, default=5.0)
    parser.add_argument("--use_default_vae", action="store_true")
    parser.add_argument("--use_chd_enc", action="store_true")
    parser.add_argument("--use_txt_enc", action="store_true")
    parser.add_argument("--custom_vae_class", type=str, default=None, help="model module path and class name. e.g. hoge.fuga.Piyo")
    parser.add_argument("--custom_vae_yaml", type=str, default=None)
    parser.add_argument("--custom_vae_ckpt", type=str, default=None)
    parser.add_argument("--is_sample", action="store_true")

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args

def save_to_checkpoint(checkpoint_dir, model, step, epoch, optimizer, is_best=False):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        # "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, "weights.pt"))
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, "best_weights.pt"))
        with open(f"{checkpoint_dir}/info.txt", "w") as f:
            f.write(str(epoch))
            f.write("\n")
            f.write(str(best_val_loss))

def parse_model_spec(model_spec: str) -> tuple[str, str]:
    """
    extract module path and class name from model specification string
    Args:
        model_spec(str): string as 'module_path.ClassName' format

    Returns:
        str: module path
        str: class name
    """
    parts = model_spec.rsplit(".", 1)  # 最後のドットで分割
    if len(parts) != 2:
        raise ValueError("model_spec must be 'module_path.ClassName' format")
    return parts[0], parts[1]  # (モジュールパス, クラス名)

def import_class(module_path, class_name):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

if __name__ == "__main__":
    args = get_args()
    output_dir = os.path.join(args.output_dir, datetime.now().strftime('%m-%d_%H%M%S'))
    log_dir = os.path.join(output_dir, "logs")
    checkpoint_dir = os.path.join(output_dir, "chkpts")
    os.makedirs(output_dir)
    os.makedirs(log_dir)
    os.makedirs(checkpoint_dir)
    print(f"result will be stored in {output_dir}")
    # パラメータの設定
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    learning_rate = args.learning_rate
    max_grad_norm = args.max_grad_norm
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    heads_num = args.heads_num
    n_layers = args.n_layers
    weight_stop_estimation = args.weight_stop_estimation
    use_default_vae = args.use_default_vae
    use_chd_enc = args.use_chd_enc
    use_txt_enc = args.use_txt_enc
    custom_vae_class = args.custom_vae_class
    custom_vae_yaml = args.custom_vae_yaml
    custom_vae_ckpt_enc = args.custom_vae_ckpt
    is_sample = args.is_sample

    debug = args.debug

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(f"{output_dir}/params.json", "w") as params_file:
        json.dump(vars(args), params_file)
    with open(f"{output_dir}/params.yaml", "w") as f:
        yaml.dump(vars(args), f)

    if use_default_vae:
        # use polyffusion pretrained vae
        if not (use_chd_enc or use_txt_enc):
            raise ValueError("use_chd_enc or use_txt_enc must be True")

        d_model = 0
        if use_chd_enc:
            # pretrained encoderの読み込み
            params_chd8bar = OmegaConf.load("polyffusion/params/sdf_chd_8bar.yaml")
            params = params_chd8bar
            chord_enc, chord_dec = load_pretrained_chd_enc_dec(
                PT_CHD_8BAR_PATH, params.chd_input_dim, params.chd_z_input_dim,
                params.chd_hidden_dim, params.chd_z_dim, params.chd_n_step
            )
            d_model += params.chd_z_dim
        else:
            chord_enc = None

        if use_txt_enc:
            params_txt = OmegaConf.load("polyffusion/params/sdf_txt.yaml")
            params = params_txt
            txt_enc = load_pretrained_txt_enc(
                PT_POLYDIS_PATH, params.txt_emb_size, params.txt_hidden_dim,
                params.txt_z_dim, params.txt_num_channel
            )
            d_model += params.txt_z_dim * 4
        else:
            txt_enc = None

        collate_fn = DefaultVAECollator(chord_enc, txt_enc)
        train_dl, val_dl = get_train_val_dataloaders_8bars(batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    else:
        # load custom vae model
        if custom_vae_yaml is None or custom_vae_ckpt_enc is None:
        # TODO: use argument `custom_vae_class`
        # if custom_vae_yaml is None or custom_vae_ckpt_enc is None or custom_vae_class is None:
            raise ValueError("custom_vae_yaml, custom_vae_ckpt_enc and custom_vae_class must be specified")

        # TODO: load and instantiate model by argument `custom_vae_class`
        # module_path, class_name = parse_model_spec(model_spec=custom_vae_class)
        # vae = import_class(module_path, class_name)(ARGS)

        with open(custom_vae_yaml, 'r', encoding='utf-8') as file:
            vae_yaml = yaml.safe_load(file)

        # TODO: ここをinit_modelに頼らないようにする
        from vae import init_model
        vae = init_model(device, vae_yaml["chd_size"], vae_yaml["txt_size"], vae_yaml["num_channel"], vae_yaml["n_bars"])
        # TODO: 既存のCollatorの名前変えたほうが良いかも Collator -> DefaultVARCollator とか

        collate_fn = CustomVAECollator(vae, is_sample)
        train_dl, val_dl = get_train_val_dataloaders_n_bars(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn, n_bars=vae_yaml["n_bars"], debug=debug)
        d_model = vae_yaml["chd_size"] + vae_yaml["txt_size"]


    # モデルにデータを流してみる
    model = GenerativeTransformerModel(
        d_model=d_model,
        heads_num=heads_num,
        n_layers=n_layers,
        device=device,
    )
    stop_loss_function = torch.nn.BCELoss()
    sequence_loss_function = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(
            log_dir
    )

    epoch = 0
    step = 0
    best_val_loss = torch.tensor([1e10], device=device)

    model.train()
    while True:
        if epoch >= max_epoch:
            break

        # train epoch loop
        for i, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch}")):
            # warmup from https://github.com/soobinseo/Transformer-TTS.git
            # if step < 400000:
            #     adjust_learning_rate(optimizer, step + 1)
            # train step loop
            for param in model.parameters():
                param.grad = None

            cond = batch[0].to(device)
            stop = batch[1]
            stop = torch.Tensor(stop)

            # forward
            output = model(cond)
            estimated_sequence, stop_probability = output

            stop_probability = stop_probability.squeeze(2)
            stop_groundtruth = torch.nn.functional.one_hot((stop - torch.ones_like(stop)).to(torch.int64), num_classes=int(torch.max(stop))).to(torch.float32).to(device)
            stop_loss = stop_loss_function(stop_probability, stop_groundtruth)
            sequence_loss = sequence_loss_function(estimated_sequence, cond)

            loss = weight_stop_estimation * stop_loss + sequence_loss

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            grad_norm = nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )

            losses = {
                "loss": loss.item(),
                "stop_loss": stop_loss.item(),
                "sequence_loss": sequence_loss.item(),
                "grad_norm": grad_norm,
            }

            step += 1
            if step % 20 == 0:
                summary_losses = losses
                writer.add_scalars("train", summary_losses, step)
                writer.flush()


        epoch += 1

        # validation
        model.eval()

        losses = None
        # valid epoch loop
        for i, batch in enumerate(val_dl):
            # train step loop

            cond = batch[0].to(device)
            stop = batch[1]
            stop = torch.Tensor(stop)

            # forward
            output = model(cond)
            estimated_sequence, stop_probability = output

            stop_probability = stop_probability.squeeze(2)
            stop_groundtruth = torch.nn.functional.one_hot((stop - torch.ones_like(stop)).to(torch.int64), num_classes=int(torch.max(stop))).to(torch.float32).to(device)
            stop_loss = stop_loss_function(stop_probability, stop_groundtruth)
            sequence_loss = sequence_loss_function(estimated_sequence, cond)

            loss = weight_stop_estimation * stop_loss + sequence_loss

            grad_norm = nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )

            current_losses = {
                "loss": loss.item(),
                "stop_loss": stop_loss.item(),
                "sequence_loss": sequence_loss.item(),
                "grad_norm": grad_norm,
            }

            if losses is None:
                losses = current_losses
            else:
                for k, v in current_losses.items():
                    losses[k] += v
            for k, v in losses.items():
                losses[k] /= len(val_dl)

        if best_val_loss >= losses["loss"]:
            best_val_loss = losses["loss"]
            save_to_checkpoint(checkpoint_dir, model, step, epoch, optimizer, is_best=True)
        else:
            save_to_checkpoint(checkpoint_dir, model, step, epoch, optimizer, is_best=False)

        summary_losses = losses
        writer.add_scalars("valid", summary_losses, step)
        writer.flush()

        model.train()
