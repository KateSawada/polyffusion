import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, f"{os.path.dirname(__file__)}/../")
from data.dataset import PianoOrchDataset
from utils import (
    pr_mat_pitch_shift, prmat2c_to_midi_file, denormalize_prmat, chd_to_onehot,
    chd_pitch_shift, onehot_to_chd, chd_to_midi_file, estx_to_midi_file,
    pianotree_pitch_shift
)


def collate_fn(batch):
    def sample_shift():
        return np.random.choice(np.arange(-6, 6), 1)[0]

    prmat_x = []
    pnotree_x = []
    chord = []
    song_fn = []
    for b in batch:
        # b[0]: seg_pnotree_x; b[1]: seg_pnotree_y
        seg_prmat_x = b[0]
        seg_pnotree_x = b[1]
        seg_chord = b[2]

        shift = sample_shift()
        seg_prmat_x = pr_mat_pitch_shift(seg_prmat_x, shift)
        seg_pnotree_x = pianotree_pitch_shift(seg_pnotree_x, shift)
        seg_chord = chd_to_onehot(chd_pitch_shift(seg_chord, shift))

        prmat_x.append(seg_prmat_x)
        pnotree_x.append(seg_pnotree_x)
        chord.append(seg_chord)

        if len(b) > 3:
            song_fn.append(b[3])

    prmat_x = torch.Tensor(np.array(prmat_x, np.float32)).float()
    pnotree_x = torch.Tensor(np.array(pnotree_x, np.int64)).long()
    chord = torch.Tensor(np.array(chord, np.float32)).float()
    # prmat_x = prmat_x.unsqueeze(1)  # (B, 1, 128, 128)
    if len(song_fn) > 0:
        return prmat_x, pnotree_x, chord, song_fn
    else:
        return prmat_x, pnotree_x, chord


def get_train_val_dataloaders(batch_size, num_workers=4, pin_memory=True, debug=False):
    train_dataset, val_dataset = PianoOrchDataset.load_train_and_valid_sets(debug)
    train_dl = DataLoader(
        train_dataset,
        batch_size,
        True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size,
        True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_dl, val_dl


if __name__ == "__main__":
    train_dl, val_dl = get_train_val_dataloaders(16)
    print(len(train_dl))
    for batch in train_dl:
        print(len(batch))
        prmat_x, pnotree_x, chord = batch
        print(prmat_x.shape)
        print(pnotree_x.shape)
        print(chord.shape)
        prmat_x = prmat_x.cpu().numpy()
        pnotree_x = pnotree_x.cpu().numpy()
        chord = chord.cpu().numpy()
        # chord = [onehot_to_chd(onehot) for onehot in chord]
        prmat2c_to_midi_file(prmat_x, f"exp/test_x.mid")
        estx_to_midi_file(pnotree_x, f"exp/test_pnotree.mid")
        chd_to_midi_file(chord, "exp/chord.mid")
        exit(0)
