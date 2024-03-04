from __future__ import annotations
import os
from typing import Callable
import tempfile
from tqdm import tqdm
from pathlib import Path

from torch.utils.data import Dataset

from dirs import (
    POP909_DATA_DIR,
    TRAIN_SPLIT_DIR,
)
from data.datasample import DataSample
from data.midi_to_data import get_data_for_single_midi
from utils import (
    read_dict,
)


SPLIT_PICKLE = "pop909_train_valid_test.pickle"


class NBarsDataSample(Dataset):
    def __init__(self, data_samples: list[DataSample]) -> None:
        super().__init__()
        self.data_samples = data_samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        return self.data_samples[index]

    @classmethod
    def load_with_song_paths(
        cls, song_paths, data_dir=POP909_DATA_DIR, debug=False, **kwargs,
    ):
        data_samples = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for song_path in tqdm(song_paths, desc="DataSample loading"):
                mid_song_path = os.path.join(
                    data_dir,
                    song_path,
                )
                data_samples += [
                    DataSample(
                        get_data_for_single_midi(
                            mid_song_path,
                            os.path.join(temp_dir, "chords_extracted.out")
                        ),
                        kwargs["n_bars"]
                    )
                ]
        return cls(data_samples)

    @classmethod
    def load_train_and_valid_sets(cls, debug=False, **kwargs,):
        if debug:
            split = read_dict(os.path.join(TRAIN_SPLIT_DIR, "pop909_debug32.pickle"))
        else:
            split = read_dict(os.path.join(TRAIN_SPLIT_DIR, SPLIT_PICKLE))
        split = list(split)
        split[0] = list(map(lambda x: add_filename_suffix(x, "_flatten"), split[0]))
        split[1] = list(map(lambda x: add_filename_suffix(x, "_flatten"), split[1]))
        split[0] = list(map(lambda x: replace_extension(x, ".mid"), split[0]))
        split[1] = list(map(lambda x: replace_extension(x, ".mid"), split[1]))

        print("load train valid set with:", kwargs)
        return cls.load_with_song_paths(
            split[0], debug=debug, **kwargs
        ), cls.load_with_song_paths(split[1], debug=debug, **kwargs)

    @classmethod
    def load_valid_sets(cls, debug=False, **kwargs,):
        if debug:
            split = read_dict(os.path.join(TRAIN_SPLIT_DIR, "pop909_debug32.pickle"))
        else:
            split = read_dict(os.path.join(TRAIN_SPLIT_DIR, SPLIT_PICKLE))
        split = list(split)
        split[1] = list(map(lambda x: add_filename_suffix(x, "_flatten"), split[1]))
        split[1] = list(map(lambda x: replace_extension(x, ".mid"), split[1]))
        print("load valid set with:", kwargs)
        return cls.load_with_song_paths(split[1], debug=debug, **kwargs)

    @classmethod
    def load_set_with_mid_dir_path(cls, mid_dir_path, **kwargs):
        directory = Path(mid_dir_path)
        midi_file_names = [f.name for f in directory.iterdir() if f.is_file() and f.suffix == ".mid"]
        print(f"loaded midi files:", midi_file_names)
        return cls.load_with_song_paths(midi_file_names, mid_dir_path, **kwargs)

def add_filename_suffix(filename: str, suffix: str) -> str:
    """add suffix before extension

    Args:
        filename (str): filename
        suffix (str): suffix

    Returns:
        str: new filename. e.g. filename="hoge.txt", suffix="fuga" -> "hoge_fuga.txt"
    """
    return filename[:filename.rfind(".")] + suffix + filename[filename.rfind("."):]

def replace_extension(filename: str, new_extension: str):
    """
    replace the extension of a file with a new one

    Args:
        filename(str): old file name
        new_extension(str): new extension. this should start with a dot.
    Returns:
        str: new file name with the new extension
    """
    old_name = Path(filename)
    new_name = old_name.with_suffix(new_extension)
    return str(new_name)
