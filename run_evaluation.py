"""
run evaluation used in polyffusion
compare between two directories and compute the difference
"""

import argparse
import glob
import tempfile
import os
import datetime

import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KernelDensity

from polyffusion.data.datasample import DataSample
from polyffusion.data.midi_to_data import get_data_for_single_midi


def overlapping_area_between_kernel_density_pdf(pdf1: KernelDensity, pdf2: KernelDensity, x_min: float, x_max: float, pdf_sample_point_num: int) -> float:
    """compute overlapping area between two kernel density pdfs
    reference: https://rinsaka.com/python/kde.html

    Args:
        pdf1 (KernelDensity): pdf1
        pdf2 (KernelDensity): pdf2
        x_min (float): min value of pdf
        x_max (float): max value of pdf
        pdf_sample_point_num (int): number of sample points

    Returns:
        float: overlapping area
    """
    x_array = np.linspace(x_min, x_max, pdf_sample_point_num).reshape(-1, 1)
    minimums = np.minimum(np.exp(pdf1.score_samples(x_array)), np.exp(pdf2.score_samples(x_array)))
    return np.sum(minimums * (x_max - x_min) / pdf_sample_point_num)


def pianotree_to_pdf(pianotree: np.ndarray) -> tuple[KernelDensity, KernelDensity]:
    """convert pianotree to pdf through kernel density estimation
    sklearn is used as kernel density estimation implementation

    Args:
        pianotree (np.ndarray): pianotree array of the whole song. shape=(bars, timesteps, 20, 6)
            timesteps = 16

    Returns:
        tuple[KernelDensity, KernelDensity]: pdf_pitch, pdf_duration
    """
    ignore_pitch_tokens = [128, 129, 130]  # sos, eos, pad
    n_pitch_token_classes = 128
    n_duration_token_classes = 32
    timesteps = 16  # bar resolution
    max_simultaneous_notes = 20

    result_pitch = []
    result_duration = []

    for bar in range(len(pianotree)):
        for timestep in range(16):
            for simu_note in range(max_simultaneous_notes):
                if (pianotree[bar, timestep, simu_note, 0] in ignore_pitch_tokens):
                    pass
                else:
                    result_pitch += [pianotree[bar, timestep, simu_note, 0]]
                    result_duration += [int("".join(list(map(str, pianotree[bar, timestep, simu_note, 1:]))), 2)]  # convert duration binary repr into int

    pdf_pitch = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(np.array(result_pitch)[:, np.newaxis])
    pdf_duration = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(np.array(result_duration)[:, np.newaxis])

    return pdf_pitch, pdf_duration


def load_files(data_root_dir, n_bars: int) -> list[DataSample]:
    """load midi files from the data_root_dir

    Args:
        data_root_dir (pathlike): midi directory
        n_bars (int): n_bars

    Returns:
        list[DataSample]: loaded samples
    """
    song_paths = glob.glob(os.path.join(data_root_dir, "*.mid"))
    samples = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for song_path in tqdm(song_paths, desc="DataSample loading"):
            data_sample = DataSample(
                get_data_for_single_midi(
                    song_path,
                    os.path.join(temp_dir, "chords_extracted.out")
                ),
                n_bars,
            )
            samples.append(data_sample)
    return samples

def datasample_to_pianotree(data_sample: DataSample) -> np.ndarray:
    """convert DataSample to pianotree

    Args:
        data_sample (DataSample): data sample

    Returns:
        np.ndarray: pianotree array. shape=(bars, timesteps, 20, 6)
    """
    return data_sample.get_whole_song_data()[1].numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", type=str, required=True)
    parser.add_argument("--dir2", type=str, required=True)
    parser.add_argument("--n_bars", type=int, default=1)
    args = parser.parse_args()

    samples1 = load_files(args.dir1, args.n_bars)
    samples2 = load_files(args.dir2, args.n_bars)

    pianotrees1 = np.vstack(list(map(datasample_to_pianotree, samples1)))
    pianotrees2 = np.vstack(list(map(datasample_to_pianotree, samples2)))

    pitch_pdf1, duration_pdf1 = pianotree_to_pdf(pianotrees1)
    pitch_pdf2, duration_pdf2 = pianotree_to_pdf(pianotrees2)
    overlapping_area_pitch = overlapping_area_between_kernel_density_pdf(pitch_pdf1, pitch_pdf2, 0, 128, 128 * 16)
    overlapping_area_duration = overlapping_area_between_kernel_density_pdf(duration_pdf1, duration_pdf2, 0, 32, 32 * 16)

    print(f"pitch: {overlapping_area_pitch}, duration: {overlapping_area_duration}")
    with open(f"eval_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.yaml", "w") as f:
        f.write(f"pitch: {overlapping_area_pitch}\nduration: {overlapping_area_duration}")
