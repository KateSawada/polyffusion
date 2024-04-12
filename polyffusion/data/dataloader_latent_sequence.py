from typing import Callable, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from data.dataset_n_bars_midi import FixedBarsDataSample
from data.dataset import DataSampleNpz
from utils import (
    pr_mat_pitch_shift,
    pianotree_pitch_shift,
    chd_pitch_shift,
    pr_mat_pitch_shift,
    chd_to_onehot,
    onehot_to_chd,
)

def get_train_val_dataloaders_n_bars(
    batch_size, num_workers=0, pin_memory=False, debug=False, n_bars=1, collate_fn:Callable=callable, **kwargs
):
    train_dataset, val_dataset = FixedBarsDataSample.load_train_and_valid_sets(
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
        False,
        collate_fn=valid_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    print(
        f"Dataloader ready: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}, {kwargs}"
    )
    return train_dl, val_dl


class CollatorWrapper:
    def __init__(self, collator: Callable, shift: bool):
        self.collator = collator
        self.shift = shift

    def __call__(self, batch: list[DataSampleNpz]) -> Any:
        return self.collator(batch, self.shift)


class CustomVAECollator:
    def __init__(self, vae: nn.Module, is_sample: bool=True, max_len: int=16):
        """_summary_

        Args:
            vae (nn.Module): vae
            is_sample (bool, optional): sampled latent value or estimated mean of distribution. Defaults to True.
        """
        self.vae = vae
        self.is_sample = is_sample
        self.max_len = max_len

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
            batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
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

        seq_length = [self.max_len] * len(batch)

        for b in batch:
            # b[0]: seg_pnotree; b[1]: seg_pnotree_y
            seg_prmat2c = b[0]
            seg_pnotree = b[1]
            seg_chord = b[2]
            seg_prmat = b[3]

            if shift:
                shift_pitch = sample_shift()
            else:
                shift_pitch = 0

            # loop for each segment
            seg_prmat2c = pr_mat_pitch_shift(seg_prmat2c, shift_pitch)
            seg_pnotree = pianotree_pitch_shift(seg_pnotree, shift_pitch)
            seg_chord = chd_pitch_shift(seg_chord, shift_pitch)
            seg_prmat = pr_mat_pitch_shift(seg_prmat, shift_pitch)

            seg_chord = chd_to_onehot(seg_chord)

            prmat2c.append(seg_prmat2c)
            pnotree.append(seg_pnotree)
            chord.append(seg_chord)
            prmat.append(seg_prmat)

            if len(b) > 4:
                song_fn.append(b[4])

        # encode one song
        prmat2c = torch.Tensor(np.array(prmat2c)).float()
        pnotree = torch.Tensor(np.array(pnotree)).long()
        chord = torch.Tensor(np.array(chord)).float()
        prmat = torch.Tensor(np.array(prmat)).float()
        # vaeへの入力となるprmatは，(B, 16, 128)の形状を持つ必要がある
        prmat_vae_input = prmat.reshape(-1, 16, 128)
        # vaeへの入力となるchordは，(B, 4, 36)の形状を持つ必要がある
        chord_vae_input = chord.reshape(-1, 4, 36)
        vae_output = self._encode(prmat_vae_input, chord_vae_input).detach()  # (B*n_bars, 1, 256*2)
        conditions = vae_output.reshape(len(batch), -1, vae_output.shape[-1])  # (B, n_bars, 256*2)

        # pad to max_sequence_length
        conditions, mask = pad_and_create_mask(conditions)

        if len(song_fn) > 0:
            return conditions, mask, seq_length, prmat2c, pnotree, chord, prmat, song_fn
        else:
            return conditions, mask, seq_length, prmat2c, pnotree, chord, prmat


def sample_shift():
    return np.random.choice(np.arange(-6, 6), 1)[0]

def pad_and_create_mask(batch_data, pad_value=0):
    max_len = max(len(data) for data in batch_data)
    batch_size = len(batch_data)

    # パディングとマスクの初期化
    padded_data = torch.full((batch_size, max_len, batch_data[0].shape[-1]), pad_value, dtype=torch.float)
    mask = torch.zeros(batch_size, max_len)

    # データをコピーし、パディングとマスクを適用
    for i, data in enumerate(batch_data):
        padded_data[i, :len(data), :] = data.squeeze(1)
        mask[i, :len(data)] = 1  # マスクの値を1に設定

    return padded_data, mask
