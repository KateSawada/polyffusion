import torch
import torch.nn as nn

from ddpm import SequenceDenoiseDiffusion
from utils import *


class Sequence_DDPM(nn.Module):
    def __init__(
        self,
        ddpm: SequenceDenoiseDiffusion,
        params,
        max_simu_note=20,
    ):
        super(Sequence_DDPM, self).__init__()
        self.params = params
        self.ddpm = ddpm

    @classmethod
    def load_trained(cls, ddpm, chkpt_fpath, params, max_simu_note=20):
        model = cls(ddpm, params, max_simu_note)
        trained_leaner = torch.load(chkpt_fpath)
        new_state_dict = {}
        for key, value in trained_leaner["state_dict"].items():
            # "model."で始まるキーの場合、"model."を削除して新しいキーに追加
            if key.startswith('model.'):
                new_key = key[len('model.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
        return model

    def p_sample(self, xt: torch.Tensor, mask: torch.Tensor, t: torch.Tensor):
        return self.ddpm.p_sample(xt, mask, t)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        return self.ddpm.q_sample(x0, t)

    def get_loss_dict(self, batch, step):
        """
        z_y is the stuff the diffusion model needs to learn
        """
        conditions, mask, seq_length, prmat2c, pnotree, chord, prmat = batch
        return {"loss": self.ddpm.loss(conditions, mask)}
