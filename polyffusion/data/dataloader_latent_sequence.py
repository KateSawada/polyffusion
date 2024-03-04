from typing import Callable, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from data.dataset_n_bars_midi import NBarsDataSample
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
    def __init__(self, vae: nn.Module, is_sample: bool=True):
        """_summary_

        Args:
            vae (nn.Module): vae
            is_sample (bool, optional): sampled latent value or estimated mean of distribution. Defaults to True.
        """
        self.vae = vae
        self.is_sample = is_sample

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
