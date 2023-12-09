import random
import torch
import torch.nn as nn
import sys


from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np

from dl_modules.chord_dec import ChordDecoder
from dl_modules.chord_enc import RnnEncoder
from dl_modules.pianotree_dec import PianoTreeDecoder
from dl_modules.txt_enc import TextureEncoder

from dl_modules.torch_plus.train_utils import get_zs_from_dists, kl_with_normal


class DisentangleVAE(nn.Module):
    def __init__(
            self,
            chord_enc: RnnEncoder,
            chord_dec: ChordDecoder,
            texture_enc: TextureEncoder,
            decoder: PianoTreeDecoder,
    ):
        super(DisentangleVAE, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.chord_enc = chord_enc
        self.chord_dec = chord_dec
        self.texture_enc = texture_enc
        self.decoder = decoder

    @classmethod
    def load_trained(cls):
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
        dist_chd = self.chord_enc(c)
        # pr_mat = self.confuse_prmat(pr_mat)
        dist_rhy = self.texture_enc(pr_mat)
        z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], True)
        dec_z = torch.cat([z_chd, z_rhy], dim=-1)
        pitch_outs, dur_outs = self.decoder(dec_z, False, embedded_x,
                                            lengths, tfr1, tfr2)
        recon_root, recon_chroma, recon_bass = self.chord_dec(z_chd, False,
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
        n_element = recon_root.shape[0] * recon_root.shape[1]
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

    def get_loss_dict(self, batch, step, tfr, beta, weights):
        prmat2c, pnotree, chord, pr_mat, p_grid = batch
        tfr1, tfr2, tfr3 = tfr
        outputs = self.run(pnotree, chord, pr_mat, tfr1, tfr2, tfr3)
        losses = self.loss_function(pnotree, chord, *outputs, beta, weights)
        return {
            "loss": losses[0],  # NOTE: もしかしたら追加しないといけないかも
        }
