import torch

from dl_modules import ChordDecoder, ChordEncoder, PianoTreeDecoder, TextureEncoder
from .model import DisentangleVAE


def load_model(chd_size, txt_size, num_channel, n_bars, chpt) -> DisentangleVAE:
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
        model.load_state_dict(torch.load(chpt)["model"])
        return model
