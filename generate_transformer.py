import argparse
import os

import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from polyffusion.params.params_sdf_chd8bar import params as params_chd8bar
from polyffusion.params.params_sdf_txt import params as params_txt

from train_transformer import GenerativeTransformerModel


def get_json_content(json_path):
    with open(json_path, 'r') as file:
        json_content = json.load(file)
    return json_content


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--trained_dir", type=str, default="result/transformer/01-19_100238",
                        help="Directory containing chkpts/ and params.json")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    output_dir = os.path.join(args.trained_dir, "generated")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    params = get_json_content(os.path.join(args.trained_dir, "params.json"))

    d_model = 0
    if params["use_chd_enc"]:
        d_model += params_chd8bar.chd_z_dim
    if params["use_txt_enc"]:
        d_model += params_txt.txt_z_dim * 4

    model = GenerativeTransformerModel(
        d_model=d_model,
        heads_num=params["heads_num"],
        n_layers=params["n_layers"],
        dropout_rate=0,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    )
    model.load_state_dict(torch.load(os.path.join(args.trained_dir, "chkpts", "best_weights.pt"))["model"])
    model.eval()

    stop_threshold = 0.9

    for i in range(args.n_samples):
        sequence = torch.randn(1, 1, d_model).to(model.device)
        stops = []
        while True:
            dec_output_seq, dec_output_stop = model(sequence)
            sequence = torch.cat((sequence, dec_output_seq[0, -1].unsqueeze(dim=0).unsqueeze(dim=0)), dim=1)
            stops.append(dec_output_stop[0, -1].item())

            if (dec_output_stop[0, -1] > stop_threshold) or (sequence.shape[1] > 256):
                sample_output_dir = os.path.join(output_dir, str(i))
                if not os.path.exists(sample_output_dir):
                    os.makedirs(sample_output_dir)
                print(f"sample {i} sequence length: {sequence.shape[1]} stop probability: {dec_output_stop[0, -1].item()}")
                np.save(os.path.join(sample_output_dir, "latent_array.npy"), sequence.to("cpu").detach().numpy())

                fig = plt.figure(figsize=(16, 8))
                fig.suptitle(f"sample {i}")
                ax = fig.add_subplot(111)
                ax.set_title(f"stop probability")
                ax.hlines(stop_threshold, 0, len(stops), colors="red", linestyles="dashed")
                ax.set_xlim(0, len(stops))
                ax.set_ylim(0, 1)
                ax.plot(stops)
                fig.savefig(os.path.join(sample_output_dir, "stop.png"))
                break
