from data.dataloader import get_custom_train_val_dataloaders, get_train_val_dataloaders
from data.dataloader_latent_sequence import get_train_val_dataloaders_n_bars, CustomVAECollator
from ddpm import DenoiseDiffusion
from models.model_ddpm import Polyffusion_DDPM
from ddpm.unet import UNet

from polydis import load_model

from . import *


class LatentDiffusion_TrainConfig(TrainConfig):
    eps_model: UNet
    # [DDPM algorithm](index.html)
    diffusion: DenoiseDiffusion

    # Adam optimizer
    optimizer: torch.optim.Adam

    def __init__(self, params, output_dir, data_dir=None):
        super().__init__(params, None, output_dir)

        self.eps_model = UNet(
            image_channels=params.image_channels,
            n_channels=params.n_channels,
            ch_mults=params.channel_multipliers,
            is_attn=params.is_attention,
        )

        # Create [DDPM class](index.html)
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=params.n_steps,
        )

        self.model = Polyffusion_DDPM(self.diffusion, params)
        # Create dataloader
        vae = load_model(params.vae.chd_size, params.vae.txt_size, params.vae.num_channel, params.vae.n_bars, params.vae.chpt)
        collate_fn = CustomVAECollator(vae, params.vae.is_sample)
        self.train_dl, self.val_dl = get_train_val_dataloaders_n_bars(params.batch_size, n_bars=params.image_size_h // 16, num_workers=params.num_workers, pin_memory=params.pin_memory, collate_fn=collate_fn, debug=params.debug)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.eps_model.parameters(), lr=params.learning_rate
        )
