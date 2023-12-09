import torch
import sys
import os

# from stable_diffusion.model.autoencoder import Autoencoder, Encoder, Decoder
from data.dataloader import get_n_bar_vae_train_val_dataloaders
from dl_modules import ChordEncoder, ChordDecoder, TextureEncoder, PianoTreeDecoder
from models.model_disentangle_vae import DisentangleVAE
from train.scheduler import MultiTeacherForcingScheduler, ParameterScheduler, ConstantScheduler, TeacherForcingScheduler
from dl_modules.torch_plus.train_utils import kl_anealing
from . import TrainConfig

class DisentangleVAE_TrainConfig(TrainConfig):
    def __init__(self, params, output_dir) -> None:
        # Teacher-forcing rate for Chord VAE training
        tfr = params.tfr
        tfr_scheduler = MultiTeacherForcingScheduler(tfr)
        weights_scheduler = ConstantScheduler(params.weights)
        beta_scheduler = TeacherForcingScheduler(params.beta, 0., f=kl_anealing)
        params_dict = dict(
            tfr=tfr_scheduler,
            beta=beta_scheduler,
            weights=weights_scheduler,
        )
        param_scheduler = ParameterScheduler(**params_dict)

        super().__init__(params, param_scheduler, output_dir)

        self.chord_enc = ChordEncoder(
            input_dim=params.chd_input_dim,
            hidden_dim=params.chd_hidden_dim,
            z_dim=params.chd_z_dim
        )
        self.chord_dec = ChordDecoder(
            input_dim=params.chd_input_dim,
            z_input_dim=params.chd_z_input_dim,
            hidden_dim=params.chd_hidden_dim,
            z_dim=params.chd_z_dim,
            n_step=params.chd_n_step
        )
        self.texture_enc = TextureEncoder(  # TODO: PtvaeDecoderにしないといけないかも
            emb_size=params.txt_emb_size,
            hidden_dim=params.txt_hidden_dim,
            z_dim=params.txt_z_dim,
            num_channel=params.txt_num_channel,
            num_bars=params.n_bars,
        )
        self.decoder = PianoTreeDecoder(
            note_embedding=None,
            dec_dur_hid_size=params.pno_dec_dur_hid_size,
            z_size=params.pno_z_size,
            num_step=params.n_bars * params.bar_resolution
        )

        self.model = DisentangleVAE(
            chord_dec=self.chord_dec,
            chord_enc=self.chord_enc,
            texture_enc=self.texture_enc,
            decoder=self.decoder,
        ).to(self.device)

        # Create dataloader
        self.train_dl, self.val_dl = get_n_bar_vae_train_val_dataloaders(params.batch_size)
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.learning_rate
        )
