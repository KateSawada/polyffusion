"""
usage

chd and rhy swap
`$ python mix_latent.py --mode mix -l1 result/FOR_GENERATE/encoded_valid_mid_small/256+256/latent_841_flatten.mid.npz -l2 result/FOR_GENERATE/encoded_valid_mid_small/256+256/latent_679_flatten.mid.npz -o result/FOR_GENERATE/encoded_valid_mid_small/256+256/chd_841_rhy_679.npy`

"""

import argparse
import os

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='Mix latent vectors from npz files')
    parser.add_argument('--mode', '-m', type=str, default='mix',
                        choices=['mix', 'avg'],)
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output file')
    parser.add_argument('--latent1', '-l1', type=str, required=True,
                        help='npz filename of the first latent vector. If mode is mix, this latent is used as chd.')
    parser.add_argument('--latent2', '-l2', type=str, required=True,
                        help='npz filename of the second latent vector. If mode is mix, this latent is used as rhy.')
    parser.add_argument('--weight', '-w', type=float, default=0.5,
                        help='Weight of the first latent vector. This works only if mode is avg.')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    latent1 = np.load(args.latent1)
    latent2 = np.load(args.latent2)
    mixed_latent = []
    if args.mode == 'avg':
        mixed_latent.append(latent1["dist_chd_mean"] * args.weight + latent2["dist_chd_mean"] * (1 - args.weight))
        mixed_latent.append(latent1["dist_rhy_mean"] * args.weight + latent2["dist_rhy_mean"] * (1 - args.weight))
    elif args.mode == 'mix':
        mixed_latent.append(latent1["dist_chd_mean"])
        mixed_latent.append(latent2["dist_rhy_mean"])
    # cut longer one
    min_len = min([len(l) for l in mixed_latent])
    mixed_latent = [l[:min_len] for l in mixed_latent]
    mixed_latent = np.concatenate(mixed_latent, axis=2)
    mixed_latent = mixed_latent.transpose(1, 0, 2)  # (1, length, latent_dim)

    print('Mixed latent shape:', mixed_latent.shape)
    # make dirs if not exist
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    np.save(args.output, mixed_latent)
    print('Saved to', args.output)
