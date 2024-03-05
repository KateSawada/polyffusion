from argparse import ArgumentParser

from omegaconf import OmegaConf
import os
import optuna
import pickle

from train.train_autoencoder import Autoencoder_TrainConfig
from train.train_chd_8bar import Chord8bar_TrainConfig
from train.train_ddpm import DDPM_TrainConfig
from train.train_ldm import LDM_TrainConfig
from train.train_latent_diffusion import LatentDiffusion_TrainConfig

if __name__ == "__main__":
    parser = ArgumentParser(
        description="train (or resume training) a Polyffusion model"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="directory in which to store model checkpoints and training logs",
    )
    parser.add_argument(
        "--data_dir", default=None, help="directory of custom training data, in npzs"
    )
    parser.add_argument(
        "--pop909_use_track", help="which tracks to use for pop909 training"
    )
    parser.add_argument("--model", help="which model to train (autoencoder, ldm, ddpm)")
    args = parser.parse_args()

    use_track = [0, 1, 2]
    if args.pop909_use_track is not None:
        use_track = [int(x) for x in args.pop909_use_track.split(",")]

    params = OmegaConf.load(f"polyffusion/params/{args.model}.yaml")

    if args.model.startswith("sdf"):
        use_musicalion = "musicalion" in args.model
        config = LDM_TrainConfig(
            params,
            args.output_dir,
            use_musicalion,
            use_track=use_track,
            data_dir=args.data_dir,
        )
    elif args.model == "ddpm":
        config = DDPM_TrainConfig(params, args.output_dir, data_dir=args.data_dir)
    elif args.model == "autoencoder":
        config = Autoencoder_TrainConfig(
            params, args.output_dir, data_dir=args.data_dir
        )
    elif args.model == "chd_8bar":
        config = Chord8bar_TrainConfig(params, args.output_dir, data_dir=args.data_dir)
    elif args.model == "latent_diffusion":
        def print_best_params(study, trial):
            # その時点での最良のパラメータを出力
            print(f"\n\nFinished trial {trial.number}: Best params {study.best_params} with value: {study.best_value}\n\n")
            with open(os.path.join(args.output_dir, "study.pkl"), "wb") as f:
                pickle.dump(study, f)
        study = optuna.create_study()
        TRIAL_SIZE = 100

        def objective(trial):
            params.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
            params.nhead = trial.suggest_categorical("heads_num", [1, 2, 4, 8, 16])
            params.num_layers = trial.suggest_int("num_layers", 1, 16)
            params.hidden_dim = trial.suggest_int("hidden_dim", 256, 4096, log=True)
            config = LatentDiffusion_TrainConfig(
                params, os.path.join(args.output_dir, str(trial.number)), data_dir=args.data_dir
            )
            return config.train()
        study.optimize(objective, n_trials=TRIAL_SIZE, callbacks=[print_best_params])

        exit()
    else:
        raise NotImplementedError
    config.train()
