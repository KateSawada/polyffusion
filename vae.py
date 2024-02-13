"""
usage

# train
`$ python vae.py --task train`

# test with validation set
`$ python vae.py --task test --ckpt /path/to/ckpt.pt`

# test with midi files
`$ python vae.py --task test --mid_dir /path/to/midi_dir --ckpt /path/to/ckpt.pt`
"""
from __future__ import annotations
import os
import sys
from typing import Any
sys.path.append(os.path.join(os.path.dirname(__file__), "polyffusion"))
sys.path.append(os.path.join(os.path.dirname(__file__), "polyffusion", "chord_extractor"))

import argparse
from datetime import datetime
import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.distributions import Normal
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import numpy as np
import yaml


from polyffusion.train.scheduler import ParameterScheduler, TeacherForcingScheduler, ConstantScheduler
from polyffusion.polydis.model import DisentangleVAE
from polyffusion.data.midi_to_data import get_data_for_single_midi
from polyffusion.data.datasample import DataSample
from polyffusion.dirs import (
    POP909_DATA_DIR,
    TRAIN_SPLIT_DIR,
)
from polyffusion.utils import (
    read_dict,
    pr_mat_pitch_shift,
    pianotree_pitch_shift,
    chd_pitch_shift,
    pr_mat_pitch_shift,
    chd_to_onehot,
    estx_to_midi_file,
)
from polyffusion.dl_modules import ChordDecoder, ChordEncoder, PianoTreeDecoder, TextureEncoder
from polyffusion.train.scheduler import OptimizerScheduler

def kl_anealing(i, high=0.1, low=0.):
    hh = 1 - low
    ll = 1 - high
    x = 10 * (i - 0.5)
    z = 1 / (1 + np.exp(x))
    y = (hh - ll) * z + ll
    return 1 - y

# TODO: 実装
def kl_sigmoid_annealing(i, high, low, lambda_, epsilon):
    pass

class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]


class PolyDisVAE(torch.nn.Module):
    def __init__(
            self,
            device: torch.device,
            chord_encoder: ChordEncoder,
            chord_decoder: ChordDecoder,
            texture_encoder: TextureEncoder,
            pianotree_decoder: PianoTreeDecoder,
    ) -> None:
        super().__init__(device)
        self.chord_encoder = chord_encoder
        self.chord_decoder = chord_decoder
        self.texture_encoder = texture_encoder
        self.pianotree_decoder = pianotree_decoder

    def forward(self, mode, *input, **kwargs):
        if mode in ["run", 0]:
            return self.run(*input, **kwargs)
        elif mode in ['loss', 'train', 1]:
            return self.loss(*input, **kwargs)
        elif mode in ['inference', 'eval', 'val', 2]:
            return self.inference(*input, **kwargs)
        else:
            raise NotImplementedError

    def confuse_prmat(self, pr_mat):
        non_zero_ent = torch.nonzero(pr_mat.long())
        eps = torch.randint(0, 2, (non_zero_ent.size(0),))
        eps = ((2 * eps) - 1).long()
        confuse_ent = torch.clamp(non_zero_ent[:, 2] + eps, min=0, max=127)
        pr_mat[non_zero_ent[:, 0], non_zero_ent[:, 1], confuse_ent] = \
            pr_mat[non_zero_ent[:, 0], non_zero_ent[:, 1], non_zero_ent[:, 2]]
        return pr_mat

    def get_chroma(self, pr_mat):
        bs = pr_mat.size(0)
        pad = torch.zeros(bs, 32, 4).to(self.device)
        pr_mat = torch.cat([pr_mat, pad], dim=-1)
        c = pr_mat.view(bs, 32, -1, 12).contiguous()
        c = c.sum(dim=-2)  # (bs, 32, 12)
        c = c.view(bs, 8, 4, 12)
        c = c.sum(dim=-2).float()
        c = torch.log(c + 1)
        return c.to(self.device)

    def run(self, x, c, pr_mat, tfr1, tfr2, tfr3, confuse=True):
        embedded_x, lengths = self.decoder.emb_x(x)
        # cc = self.get_chroma(pr_mat)
        dist_chd = self.chd_encoder(c)
        # pr_mat = self.confuse_prmat(pr_mat)
        dist_rhy = self.rhy_encoder(pr_mat)
        z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], True)
        dec_z = torch.cat([z_chd, z_rhy], dim=-1)
        pitch_outs, dur_outs = self.decoder(dec_z, False, embedded_x,
                                            lengths, tfr1, tfr2)
        recon_root, recon_chroma, recon_bass = self.chd_decoder(z_chd, False,
                                                                tfr3, c)
        return pitch_outs, dur_outs, dist_chd, dist_rhy, recon_root, \
            recon_chroma, recon_bass

    def loss_function(self, x, c, recon_pitch, recon_dur, dist_chd,
                      dist_rhy, recon_root, recon_chroma, recon_bass,
                      beta, weights, weighted_dur=False):
        recon_loss, pl, dl = self.decoder.recon_loss(x, recon_pitch, recon_dur,
                                                     weights, weighted_dur)
        kl_loss, kl_chd, kl_rhy = self.kl_loss(dist_chd, dist_rhy)
        chord_loss, root, chroma, bass = self.chord_loss(c, recon_root,
                                                         recon_chroma,
                                                         recon_bass)
        loss = recon_loss + beta * kl_loss + chord_loss
        return loss, recon_loss, pl, dl, kl_loss, kl_chd, kl_rhy, chord_loss, \
               root, chroma, bass

    def chord_loss(self, c, recon_root, recon_chroma, recon_bass):
        loss_fun = nn.CrossEntropyLoss()
        root = c[:, :, 0: 12].max(-1)[-1].view(-1).contiguous()
        chroma = c[:, :, 12: 24].long().view(-1).contiguous()
        bass = c[:, :, 24:].max(-1)[-1].view(-1).contiguous()

        recon_root = recon_root.view(-1, 12).contiguous()
        recon_chroma = recon_chroma.view(-1, 2).contiguous()
        recon_bass = recon_bass.view(-1, 12).contiguous()
        root_loss = loss_fun(recon_root, root)
        chroma_loss = loss_fun(recon_chroma, chroma)
        bass_loss = loss_fun(recon_bass, bass)
        chord_loss = root_loss + chroma_loss + bass_loss
        return chord_loss, root_loss, chroma_loss, bass_loss

    def kl_loss(self, *dists):
        # kl = kl_with_normal(dists[0])
        kl_chd = kl_with_normal(dists[0])
        kl_rhy = kl_with_normal(dists[1])
        kl_loss = kl_chd + kl_rhy
        return kl_loss, kl_chd, kl_rhy

    def loss(self, x, c, pr_mat, tfr1=0., tfr2=0., tfr3=0.,
             beta=0.1, weights=(1, 0.5)):
        outputs = self.run(x, c, pr_mat, tfr1, tfr2, tfr3)
        loss = self.loss_function(x, c, *outputs, beta, weights)
        return loss

    def inference_encode(self, pr_mat, c):
        self.eval()
        with torch.no_grad():
            dist_chd = self.chd_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
        return dist_chd, dist_rhy

    def inference_decode(self, z_chd, z_rhy):
        self.eval()
        with torch.no_grad():
            dec_z = torch.cat([z_chd, z_rhy], dim=-1)
            pitch_outs, dur_outs = self.decoder(dec_z, True, None,
                                                None, 0., 0.)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
        return est_x

    def inference(self, pr_mat, c, sample):
        self.eval()
        with torch.no_grad():
            dist_chd = self.chd_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
            z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], sample)
            dec_z = torch.cat([z_chd, z_rhy], dim=-1)
            pitch_outs, dur_outs = self.decoder(dec_z, True, None,
                                                None, 0., 0.)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
        return est_x

    def swap(self, pr_mat1, pr_mat2, c1, c2, fix_rhy, fix_chd):
        pr_mat = pr_mat1 if fix_rhy else pr_mat2
        c = c1 if fix_chd else c2
        est_x = self.inference(pr_mat, c, sample=False)
        return est_x

    def posterior_sample(self, pr_mat, c, scale=None, sample_chd=True,
                         sample_txt=True):
        if scale is None and sample_chd and sample_txt:
            est_x = self.inference(pr_mat, c, sample=True)
        else:
            dist_chd, dist_rhy = self.inference_encode(pr_mat, c)
            if scale is not None:
                mean_chd = dist_chd.mean
                mean_rhy = dist_rhy.mean
                # std_chd = torch.ones_like(dist_chd.mean) * scale
                # std_rhy = torch.ones_like(dist_rhy.mean) * scale
                std_chd = dist_chd.scale * scale
                std_rhy = dist_rhy.scale * scale
                dist_rhy = Normal(mean_rhy, std_rhy)
                dist_chd = Normal(mean_chd, std_chd)
            z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], True)
            if not sample_chd:
                z_chd = dist_chd.mean
            if not sample_txt:
                z_rhy = dist_rhy.mean
            est_x = self.inference_decode(z_chd, z_rhy)
        return est_x

    def prior_sample(self, x, c, sample_chd=False, sample_rhy=False,
                     scale=1.):
        dist_chd, dist_rhy = self.inference_encode(x, c)
        mean = torch.zeros_like(dist_rhy.mean)
        loc = torch.ones_like(dist_rhy.mean) * scale
        if sample_chd:
            dist_chd = Normal(mean, loc)
        if sample_rhy:
            dist_rhy = Normal(mean, loc)
        z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], True)
        return self.inference_decode(z_chd, z_rhy)

    def gt_sample(self, x):
        out = x[:, :, 1:].numpy()
        return out

    def interp(self, pr_mat1, c1, pr_mat2, c2, interp_chd=False,
               interp_rhy=False, int_count=10):
        dist_chd1, dist_rhy1 = self.inference_encode(pr_mat1, c1)
        dist_chd2, dist_rhy2 = self.inference_encode(pr_mat2, c2)
        [z_chd1, z_rhy1, z_chd2, z_rhy2] = \
            get_zs_from_dists([dist_chd1, dist_rhy1, dist_chd2, dist_rhy2],
                              False)
        if interp_chd:
            z_chds = self.interp_z(z_chd1, z_chd2, int_count)
        else:
            z_chds = z_chd1.unsqueeze(1).repeat(1, int_count, 1)
        if interp_rhy:
            z_rhys = self.interp_z(z_rhy1, z_rhy2, int_count)
        else:
            z_rhys = z_rhy1.unsqueeze(1).repeat(1, int_count, 1)
        bs = z_chds.size(0)
        z_chds = z_chds.view(bs * int_count, -1).contiguous()
        z_rhys = z_rhys.view(bs * int_count, -1).contiguous()
        estxs = self.inference_decode(z_chds, z_rhys)
        return estxs.reshape((bs, int_count, 32, 15, -1))

    def interp_z(self, z1, z2, int_count=10):
        z1 = z1.numpy()
        z2 = z2.numpy()
        zs = torch.stack([self.interp_path(zz1, zz2, int_count)
                          for zz1, zz2 in zip(z1, z2)], dim=0)
        return zs

    def interp_path(self, z1, z2, interpolation_count=10):
        result_shape = z1.shape
        z1 = z1.reshape(-1)
        z2 = z2.reshape(-1)

        def slerp2(p0, p1, t):
            omega = np.arccos(
                np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
            so = np.sin(omega)
            return np.sin((1.0 - t) * omega)[:, None] / so * p0[
                None] + np.sin(
                t * omega)[:, None] / so * p1[None]

        percentages = np.linspace(0.0, 1.0, interpolation_count)

        normalized_z1 = z1 / np.linalg.norm(z1)
        normalized_z2 = z2 / np.linalg.norm(z2)
        dirs = slerp2(normalized_z1, normalized_z2, percentages)
        length = np.linspace(np.log(np.linalg.norm(z1)),
                             np.log(np.linalg.norm(z2)),
                             interpolation_count)
        out = (dirs * np.exp(length[:, None])).reshape(
            [interpolation_count] + list(result_shape))
        # out = np.array([(1 - t) * z1 + t * z2 for t in percentages])
        return torch.from_numpy(out).to(self.device).float()

    @staticmethod
    def init_model(
        device: torch.device,
        chd_size: int=256,
        txt_size: int=256,
        num_channel: int=10,
        n_bars: int=1,
    ) -> PolyDisVAE:
        chord_encoder = ChordEncoder(36, 256, chd_size)
        chord_decoder = ChordDecoder(z_dim=chd_size, n_step=n_bars*4)
        texture_encoder = TextureEncoder(256, 512, txt_size, num_channel, n_bars*16)
        pianotree_decoder = PianoTreeDecoder(note_embedding=None, dec_dur_hid_size=64, z_size=chd_size+txt_size, num_step=16*n_bars)

        model = PolyDisVAE(
            device,
            chord_encoder,
            chord_decoder,
            texture_encoder,
            pianotree_decoder
        )
        return model

def init_model(
        device: torch.device,
        chd_size: int=256,
        txt_size: int=256,
        num_channel: int=10,
        n_bars: int=1,
    ) -> DisentangleVAE:
        chord_encoder = ChordEncoder(36, 256, chd_size)
        chord_decoder = ChordDecoder(z_dim=chd_size, n_step=n_bars*4)
        texture_encoder = TextureEncoder(256, 512, txt_size, num_channel, n_bars*16)
        pianotree_decoder = PianoTreeDecoder(note_embedding=None, dec_dur_hid_size=64, z_size=chd_size+txt_size, num_step=16*n_bars)

        model = DisentangleVAE(
            "polydis_vae",
            chord_encoder,
            texture_encoder,
            pianotree_decoder,
            chord_decoder,
        )
        return model.to(device)

def add_filename_suffix(filename: str, suffix: str) -> str:
    """add suffix before extension

    Args:
        filename (str): filename
        suffix (str): suffix

    Returns:
        str: new filename. e.g. filename="hoge.txt", suffix="fuga" -> "hoge_fuga.txt"
    """
    return filename[:filename.rfind(".")] + suffix + filename[filename.rfind("."):]

def replace_extension(filename: str, new_extension: str):
    """
    replace the extension of a file with a new one

    Args:
        filename(str): old file name
        new_extension(str): new extension. this should start with a dot.
    Returns:
        str: new file name with the new extension
    """
    old_name = Path(filename)
    new_name = old_name.with_suffix(new_extension)
    return str(new_name)


class NBarsDataSample(Dataset):
    def __init__(self, data_samples: list[DataSample]) -> None:
        super().__init__()
        self.data_samples = data_samples

        self.lgths = np.array([len(d) for d in self.data_samples], dtype=np.int64)
        self.lgth_cumsum = np.cumsum(self.lgths)

    def __len__(self):
        return self.lgth_cumsum[-1]

    def __getitem__(self, index):
        # song_no is the smallest id that > dataset_item
        song_no = np.where(self.lgth_cumsum > index)[0][0]
        song_item = index - np.insert(self.lgth_cumsum, 0, 0)[song_no]

        song_data = self.data_samples[song_no]
        return song_data[song_item]

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

# NOTE: このvaeでは使わないのに，勢い余って実装してしまった．
class WholeSongDataSample(Dataset):
    def __init__(self, data_samples: list[DataSample]) -> None:
        super().__init__()
        self.data_samples = data_samples

    @classmethod
    def load_with_song_paths(
        cls, song_paths, data_dir=POP909_DATA_DIR, debug=False, **kwargs,
    ):
        data_samples = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for song_path in tqdm(song_paths, desc="DataSample loading"):
                mid_song_path = os.path.join(
                    data_dir,
                    os.path.dirname(song_path),
                    os.path.splitext(os.path.basename(song_path))[0] + "_flatten.mid"
                )
                data_samples += [
                    DataSample(
                        get_data_for_single_midi(
                            mid_song_path,
                            os.path.join(temp_dir, "chords_extracted.out")
                        ),
                        debug,
                    )
                ]
        return cls(data_samples)

    @classmethod
    def load_train_and_valid_sets(cls, debug=False, **kwargs,):
        if debug:
            split = read_dict(os.path.join(TRAIN_SPLIT_DIR, "pop909_debug32.pickle"))
        else:
            split = read_dict(os.path.join(TRAIN_SPLIT_DIR, "pop909.pickle"))
        print("load train valid set with:", kwargs)
        return cls.load_with_song_paths(
            split[0], debug=debug, **kwargs
        ), cls.load_with_song_paths(split[1], debug=debug, **kwargs)

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, index: int) -> DataSample:
        """returns prmat.,pnotree, chord, prmat.

        Args:
            index (int): index

        Returns:
            DataSample
        """
        return self.data_samples[index]

def sample_shift():
    return np.random.choice(np.arange(-6, 6), 1)[0]

def collate_fn(batch, shift):
    prmat2c = []
    pnotree = []
    chord = []
    prmat = []
    song_fn = []
    for b in batch:
        # b[0]: seg_pnotree; b[1]: seg_pnotree_y
        seg_prmat2c = b[0]
        seg_pnotree = b[1]
        seg_chord = b[2]
        seg_prmat = b[3]

        if shift:
            shift_pitch = sample_shift()
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

    prmat2c = torch.Tensor(np.array(prmat2c, np.float32)).float()
    pnotree = torch.Tensor(np.array(pnotree, np.int64)).long()
    chord = torch.Tensor(np.array(chord, np.float32)).float()
    prmat = torch.Tensor(np.array(prmat, np.float32)).float()
    # prmat = prmat.unsqueeze(1)  # (B, 1, 128, 128)
    if len(song_fn) > 0:
        return prmat2c, pnotree, chord, prmat, song_fn
    else:
        return prmat2c, pnotree, chord, prmat

def get_train_val_dataloaders(
    batch_size, num_workers=0, pin_memory=False, debug=False, **kwargs
):
    train_dataset, val_dataset = NBarsDataSample.load_train_and_valid_sets(
        debug=debug, **kwargs
    )
    train_collator = Collator(is_shift=True)
    val_collator = Collator(is_shift=False)
    train_dl = DataLoader(
        train_dataset,
        batch_size,
        True,
        collate_fn=train_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size,
        False,
        collate_fn=val_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print(
        f"Dataloader ready: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}, train_segments={len(train_dataset)}, val_segments={len(val_dataset)} {kwargs}"
    )
    return train_dl, val_dl

def get_val_dataloader(
    batch_size, num_workers=0, pin_memory=False, debug=False, **kwargs
):
    val_dataset = NBarsDataSample.load_valid_sets(
        debug=debug, **kwargs
    )
    val_collator = Collator(is_shift=False)
    val_dl = DataLoader(
        val_dataset,
        batch_size,
        False,
        collate_fn=val_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print(
        f"Dataloader ready: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}, val_segments={len(val_dataset)} {kwargs}"
    )
    return val_dl

def get_mid_dataloader(
    batch_size, num_workers=0, pin_memory=False, mid_dir=None, **kwargs
):
    mid_dataset = NBarsDataSample.load_set_with_mid_dir_path(
        mid_dir, **kwargs
    )
    mid_collator = Collator(is_shift=False)
    mid_loader = DataLoader(
        mid_dataset,
        batch_size,
        False,
        collate_fn=mid_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print(
        f"Dataloader ready: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}, mid_segments={len(mid_dataset)} {kwargs}"
    )
    return mid_loader

class Collator(object):
    def __init__(self, is_shift=False):
        self.is_shift = is_shift

    def __call__(self, batch) -> Any:
        return collate_fn(batch, shift=self.is_shift)

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

def _accumulate_loss_dic(writer_names, loss_dic, loss_items):
    assert len(writer_names) == len(loss_items)
    for key, val in zip(writer_names, loss_items):
        loss_dic[key] += val.item()
    return loss_dic

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="train", help="train or test")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--output_dir", type=str, default="result/polydis_vae")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_bars", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=10)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", type=bool, default=True)

    # parser.add_argument("--max_simu_note", type=int, default=20)
    # parser.add_argument("--max_pitch", type=int, default=127)
    # parser.add_argument("--min_pitch", type=int, default=0)
    # parser.add_argument("--pitch_sos", type=int, default=128)
    # parser.add_argument("--pitch_eos", type=int, default=129)
    # parser.add_argument("--pitch_pad", type=int, default=130)
    # parser.add_argument("--dur_pad", type=int, default=2)
    # parser.add_argument("--dur_width", type=int, default=5)
    # parser.add_argument("--num_step", type=int, default=16)
    # parser.add_argument("--note_emb_size", type=int, default=128)
    # parser.add_argument("--enc_notes_hid_size", type=int, default=256)
    # parser.add_argument("--enc_time_hid_size", type=int, default=512)
    # parser.add_argument("--z_size", type=int, default=512)
    # parser.add_argument("--dec_emb_hid_size", type=int, default=128)
    # parser.add_argument("--dec_time_hid_size", type=int, default=1024)
    # parser.add_argument("--dec_notes_hid_size", type=int, default=512)
    # parser.add_argument("--dec_z_in_size", type=int, default=256)
    # parser.add_argument("--dec_dur_hid_size", type=int, default=64)
    parser.add_argument("--chd_size", type=int, default=256)
    parser.add_argument("--txt_size", type=int, default=256)
    parser.add_argument("--num_channel", type=int, default=10)

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint file(.pt) path",
        required=False,
    )
    parser.add_argument(
        "--mid_dir",
        type=str,
        default=None,
        help="midi directory",
        required=False,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # for i in range(10):
    #     filename = f"data/POP909_4_bin_pnt_8bar/{(i+1):03d}_flatten.mid"
    #     data = get_data_for_single_midi(filename, "exp/chords_extracted_inpaint.out")


    #     data_sample = DataSample(data, n_bars=1)
    #     print("\n" * 4)
    #     break

    # from polyffusion.data.midi_to_data import get_chord_matrix
    # import numpy as np
    # from polyffusion.utils import chd_to_onehot

    # print(np.array(get_chord_matrix("exp/chords_extracted_inpaint.out")))
    # print(data_sample._get_item_by_db(0)[2])
    # print(data_sample._get_item_by_db(0)[2].shape)
    # print(len(data_sample))

    # for i in range(1):
    #     prmat2c, pnotree, chord, prmat = data_sample[i]
    #     np.save(f"tmp/8bar/{i}_prmat2c.npy", prmat2c)
    #     np.save(f"tmp/8bar/{i}_pnotree.npy", pnotree)
    #     np.save(f"tmp/8bar/{i}_chord.npy", chord)
    #     np.save(f"tmp/8bar/{i}_prmat.npy", prmat)

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
    n_bars = args.n_bars
    max_epoch = args.max_epoch
    learning_rate = args.learning_rate
    max_grad_norm = args.max_grad_norm
    num_workers = args.num_workers
    pin_memory = args.pin_memory

    chd_size = args.chd_size
    txt_size = args.txt_size
    num_channel = args.num_channel

    ckpt = args.ckpt
    mid_dir = args.mid_dir

    clip = 1
    tf_rates = [(0.6, 0), (0.5, 0), (0.5, 0)]


    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    with open(f"{output_dir}/params.json", "w") as params_file:
        json.dump(vars(args), params_file)
    with open(f"{output_dir}/params.yaml", "w") as f:
        yaml.dump(vars(args), f)

    if args.task == "train":
        train_dl, val_dl = get_train_val_dataloaders(**vars(args))

        model = init_model(device, chd_size, txt_size, num_channel, n_bars)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = MinExponentialLR(optimizer, gamma=0.9999, minimum=1e-5)
        opt_scheduler = OptimizerScheduler(optimizer, scheduler, clip)

        tfr1_scheduler = TeacherForcingScheduler(*tf_rates[0])
        tfr2_scheduler = TeacherForcingScheduler(*tf_rates[1])
        tfr3_scheduler = TeacherForcingScheduler(*tf_rates[2])
        weights = [1, 0.5]
        weights_scheduler = ConstantScheduler(weights)
        beta = [0.1, 0.0001]  # TODO: これの良い値を考える
        beta_scheduler = TeacherForcingScheduler(*beta, f=kl_anealing)
        params_dic = dict(tfr1=tfr1_scheduler, tfr2=tfr2_scheduler,
                        tfr3=tfr3_scheduler,
                        beta=beta_scheduler, weights=weights_scheduler)
        param_scheduler = ParameterScheduler(**params_dic)

        writer_names = ['loss', 'recon_loss', 'pl', 'dl', 'kl_loss', 'kl_chd',
                    'kl_rhy', 'chord_loss', 'root_loss', 'chroma_loss', 'bass_loss']
        writer = SummaryWriter(
                log_dir
        )

        epoch = 0
        step = 0
        best_val_loss = torch.tensor([1e10], device=device)

        model.train()
        param_scheduler.train()

        while True:

            if epoch >= max_epoch:
                break

            losses = None
            # train epoch loop
            for i, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch}")):
                # warmup from https://github.com/soobinseo/Transformer-TTS.git
                # if step < 400000:
                #     adjust_learning_rate(optimizer, step + 1)
                # train step loop
                for param in model.parameters():
                    param.grad = None

                prmat2c, pnotree, chord, prmat = batch
                input_params = param_scheduler.step()

                outputs = model.loss(pnotree.to(device), chord.to(device), prmat.to(device), None, **input_params)  # None is alternative to dt_x
                loss = outputs[0]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    opt_scheduler.clip)
                opt_scheduler.step()
                outputs = dict(zip(writer_names, map(lambda x: x.detach().cpu().numpy().copy(), outputs)))

                if losses is None:
                    losses = outputs
                else:
                    for k, v in outputs.items():
                        losses[k] += v

                if step % 100 == 0:
                    writer.add_scalars("train_step", outputs, step)
                step += 1

            for k, v in losses.items():
                losses[k] /= len(train_dl)

            writer.add_scalars("train", losses, step)
            writer.flush()
            epoch += 1

            # validation
            model.eval()
            param_scheduler.eval()

            losses = None
            # valid epoch loop
            for i, batch in enumerate(val_dl):
                # val step loop
                prmat2c, pnotree, chord, prmat = batch
                input_params = param_scheduler.step()

                with torch.no_grad():
                    outputs = model.loss(pnotree.to(device), chord.to(device), prmat.to(device), None, **input_params)

                outputs = dict(zip(writer_names, map(lambda x: x.detach().cpu().numpy().copy(), outputs)))
                if losses is None:
                    losses = outputs
                else:
                    for k, v in outputs.items():
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

    elif args.task == "test":
        # reconstruct
        if (ckpt is None):
            raise ValueError("checkpoint file is not specified.")

        model = init_model(device, chd_size, txt_size, num_channel, n_bars)
        state_dict = torch.load(ckpt)
        model.load_state_dict(state_dict["model"])
        model.eval()

        if (mid_dir is None):
            # generate with val_dl
            val_dl = get_val_dataloader(**vars(args))
            for i, batch in enumerate(tqdm(val_dl)):
                # val step loop
                prmat2c, pnotree, chord, prmat = batch

                with torch.no_grad():
                    outputs = model.run(pnotree.to(device), chord.to(device), prmat.to(device), 0.0, 0.0, 0.0)
                (
                    pitch_outs,
                    dur_outs,
                    dist_chd,
                    dist_rhy,
                    recon_root,
                    recon_chroma,
                    recon_bass,
                ) = outputs

                est_x, _, _ = model.decoder.output_to_numpy(pitch_outs, dur_outs)
                estx_to_midi_file(est_x, os.path.join(output_dir, f"est_{i}.mid"))
        else:
            # reconstruct midi files in mid_dir
            loader = get_mid_dataloader(**vars(args))
            for i, batch in enumerate(tqdm(loader)):
                prmat2c, pnotree, chord, prmat = batch

                with torch.no_grad():
                    outputs = model.run(pnotree.to(device), chord.to(device), prmat.to(device), 0.0, 0.0, 0.0)
                (
                    pitch_outs,
                    dur_outs,
                    dist_chd,
                    dist_rhy,
                    recon_root,
                    recon_chroma,
                    recon_bass,
                ) = outputs

                est_x, _, _ = model.decoder.output_to_numpy(pitch_outs, dur_outs)
                estx_to_midi_file(est_x, os.path.join(output_dir, f"est_{i}.mid"))
