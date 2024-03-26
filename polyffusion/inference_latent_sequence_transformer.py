import os
import pickle
from argparse import ArgumentParser
from datetime import datetime
from os.path import join
from pathlib import Path
import uuid

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

from data.dataset import DataSampleNpz
from ddpm import SequenceDenoiseDiffusion
from ddpm.transformer import TransformerEncoderModel
from ddpm.utils import gather
from dirs import *
from models.model_latent_diffusion_transformer import Sequence_DDPM
from utils import prmat2c_to_midi_file, show_image
from polydis import DisentangleVAE, load_model


class Configs:
    # U-Net model for $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
    eps_model: TransformerEncoderModel
    # [DDPM algorithm](index.html)
    diffusion: SequenceDenoiseDiffusion

    guide_model: DisentangleVAE

    # Adam optimizer
    optimizer: torch.optim.Adam

    def __init__(self, params, model_dir, chkpt_name="weights_best.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)
        self.eps_model = TransformerEncoderModel(
            input_dim=params.input_dim,
            hidden_dim=params.hidden_dim,
            num_layers=params.num_layers,
            nhead=params.nhead,
            max_len=params.max_len,
            pe_n_dim_divide=params.pe_n_dim_divide,
            pe_strength=params.pe_strength,
        ).to(self.device)

        # Create [DDPM class](index.html)
        self.diffusion = SequenceDenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=params.n_steps,
            # device=self.device,
        ).to(self.device)

        self.guide_model = load_model(params.vae.chd_size, params.vae.txt_size, params.vae.num_channel, params.vae.n_bars, params.vae.chpt).to(self.device)

        self.model = Sequence_DDPM.load_trained(
            self.diffusion, os.path.join(model_dir, "chkpts", chkpt_name), params
        ).to(self.device)

        # self.song_fn, self.pnotree, _ = choose_song_from_val_dl()

        self.max_len = params.max_len
        self.input_dim = params.input_dim
        self.n_steps = params.n_steps
        # $\beta_t$
        self.beta = self.diffusion.beta
        # $\alpha_t$
        self.alpha = self.diffusion.alpha.to(self.device)
        # $\bar\alpha_t$
        self.alpha_bar = self.diffusion.alpha_bar.to(self.device)
        # $\bar\alpha_{t-1}$
        alpha_bar_tm1 = torch.cat([self.alpha_bar.new_ones((1,)), self.alpha_bar[:-1]]).to(self.device)

        # $\tilde\beta_t$
        self.beta_tilde = self.beta * (1 - alpha_bar_tm1) / (1 - self.alpha_bar).to(self.device)
        # $$\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$$
        self.mu_tilde_coef1 = self.beta * (alpha_bar_tm1**0.5) / (1 - self.alpha_bar)
        # $$\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1}}{1-\bar\alpha_t}$$
        self.mu_tilde_coef2 = (
            (self.alpha**0.5) * (1 - alpha_bar_tm1) / (1 - self.alpha_bar)
        )
        # $\sigma^2 = \beta$
        self.sigma2 = self.beta

        self.params = params

    def classifier_guidance_step(self, xt: torch.Tensor):
        shape = xt.shape
        xt_input = xt.reshape(-1, shape[-1]).clone().detach()
        with torch.set_grad_enabled(True):
            xt_input.requires_grad_()
            pitch, dur = self.guide_model.decoder(xt_input, True, None, None, 0.0, 0.0)
            (pitch.mean() + dur.mean()).backward()
            xt_grad = xt_input.grad
            xt = xt - xt_grad.reshape(*shape)
        return xt

    def _sample_x0(self, xt: torch.Tensor, n_steps: int, show_img=False):
        """
        #### Sample an image using $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        * `xt` is $x_t$
        * `n_steps` is $t$
        """

        # Number of samples
        n_samples = xt.shape[0]
        # create mask tensor that contains no masking(1)
        print(xt.shape)
        mask = torch.zeros((n_samples, xt.shape[2])).to(self.device)
        # Iterate until $t$ steps
        for t_ in tqdm(range(n_steps), desc="Sampling"):
            t = n_steps - t_ - 1
            # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
            xt = self.model.p_sample(xt, mask, xt.new_full((n_samples,), t, dtype=torch.long))
            xt = self.classifier_guidance_step(xt)
            if t_ % 100 == 0 or (t_ >= 900 and t_ % 25 == 0):
                if show_img:
                    show_image(xt, f"exp/x{t}.png")
                if n_samples > 1:
                    prmat = xt.detach().squeeze().cpu().numpy()
                else:
                    prmat = xt.detach().cpu().numpy()
                if show_img:
                    show_image(xt, f"exp/x{t}.png")
                    prmat2c_to_midi_file(prmat, f"exp/x{t + 1}.mid")

        # Return $x_0$
        return xt

    def sample(
        self, n_samples: int = 1, init_cond=None, init_step=None, show_img=False
    ):
        """
        #### Generate images
        """
        # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
        if init_cond is not None:
            init_cond = init_cond.to(self.device)
            assert init_step is not None
            xt = self.model.q_sample(
                init_cond,
                init_cond.new_full((init_cond.shape[0],), init_step, dtype=torch.long),
            )
        else:
            xt = torch.randn(
                [n_samples, 1, self.max_len, self.input_dim],
                device=self.device,
            )

        init_step = init_step or self.n_steps
        # $$x_0 \sim \textcolor{lightgreen}{p_\theta}(x_0|x_t)$$
        x0 = self._sample_x0(xt, init_step, show_img=show_img)

        return x0

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, eps_theta: torch.Tensor):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
        """
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha**0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var**0.5) * eps

    def predict(
        self,
        n_samples: int = 16,
        init_cond=False,
        init_step=None,
        output_dir="exp",
        show_img=False,
    ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.eval()
        with torch.no_grad():
            if not init_cond:
                x0 = self.sample(n_samples, show_img=show_img).squeeze(1)
                if show_img:
                    show_image(x0, os.path.join(output_dir, "x0.png"))
                if n_samples > 1:
                    prmat = x0.detach().squeeze().cpu().numpy()
                else:
                    prmat = x0.detach().cpu().numpy()
                output_stamp = f"ddpm_prmat2c_[uncond]_{datetime.now().strftime('%y-%m-%d_%H%M%S')}-{uuid.uuid1()}"
                np.save(os.path.join(output_dir, f"{output_stamp}.npy"), prmat)
                return x0
            else:
                song_fn, x_init, _ = choose_song_from_val_dl()
                x0 = self.sample(
                    n_samples, init_cond=x_init, init_step=init_step, show_img=show_img
                )
                if show_img:
                    show_image(x0, os.path.join(output_dir, "x0.png"))
                if n_samples > 1:
                    prmat = x0.detach().squeeze().cpu().numpy()
                else:
                    prmat = x0.detach().cpu().numpy()
                output_stamp = f"ddpm_prmat2c_init_[{song_fn}]_{datetime.now().strftime('%y-%m-%d_%H%M%S')}-{uuid.uuid1()}"
                prmat2c_to_midi_file(
                    prmat, os.path.join(output_dir, f"{output_stamp}.mid")
                )
                return x0


def choose_song_from_val_dl():
    split_fpath = join(TRAIN_SPLIT_DIR, "musicalion.pickle")
    with open(split_fpath, "rb") as f:
        split = pickle.load(f)
    print(split[1])
    num = int(input("choose one:"))
    song_fn = split[1][num]
    print(song_fn)

    song = DataSampleNpz(song_fn)
    _, _, _, prmat = song.get_whole_song_data()
    prmat_np = prmat.squeeze().cpu().numpy()
    prmat2c_to_midi_file(prmat_np, "exp/origin.mid")
    return song_fn, prmat, prmat


if __name__ == "__main__":
    parser = ArgumentParser(description="inference a Polyffusion model")
    parser.add_argument(
        "--model_dir", help="directory in which trained model checkpoints are stored"
    )
    parser.add_argument(
        "--length", type=int, default=1, help="number of 8 bars to generate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="exp", help="output directory"
    )
    parser.add_argument(
        "--num_generate", type=int, default=1, help="number of inferences"
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="whether to generate progress images and midis",
    )
    parser.add_argument(
        "--chkpt_name",
        default="weights_best.pt",
        help="which specific checkpoint to use (default: weights_best.pt)",
    )
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except FileExistsError:
            # avoid error ocurred when multiple processes try to create the same directory
            pass
    params = OmegaConf.load(os.path.join(args.model_dir, "params.yaml"))
    config = Configs(params, args.model_dir, args.chkpt_name)
    for i in range(args.num_generate):
        config.predict(
            n_samples=args.length,
            init_cond=False,
            init_step=100,
            output_dir=args.output_dir,
            show_img=args.show_progress,
        )
