from data.dataloader import get_custom_train_val_dataloaders, get_train_val_dataloaders
from data.dataloader_latent_sequence import get_train_val_dataloaders_n_bars, CustomVAECollator
from ddpm import SequenceDenoiseDiffusion
from ddpm.transformer import TransformerEncoderModel
from models.model_latent_diffusion_transformer import Sequence_DDPM
from polydis import load_model

from . import *


class LatentDiffusion_TrainConfig(TrainConfig):
    eps_model: TransformerEncoderModel
    # [DDPM algorithm](index.html)
    diffusion: SequenceDenoiseDiffusion

    # Adam optimizer
    optimizer: torch.optim.Adam

    def __init__(self, params, output_dir, data_dir=None):
        super().__init__(params, None, output_dir)

        self.eps_model = TransformerEncoderModel(
            input_dim=params.input_dim,
            hidden_dim=params.hidden_dim,
            num_layers=params.num_layers,
            nhead=params.nhead,
            max_len=params.max_len,
            pe_n_dim_divide=params.pe_n_dim_divide,
            pe_strength=params.pe_strength,
        )

        # Create [DDPM class](index.html)
        self.diffusion = SequenceDenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=params.n_steps,
        )

        self.model = Sequence_DDPM(self.diffusion, params)
        # Create dataloader
        vae = load_model(params.vae.chd_size, params.vae.txt_size, params.vae.num_channel, params.vae.n_bars, params.vae.chpt)
        collate_fn = CustomVAECollator(vae, params.vae.is_sample)
        self.train_dl, self.val_dl = get_train_val_dataloaders_n_bars(params.batch_size, n_bars=params.max_len, num_workers=params.num_workers, pin_memory=params.pin_memory, collate_fn=collate_fn, debug=params.debug)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.eps_model.parameters(), lr=params.learning_rate
        )
