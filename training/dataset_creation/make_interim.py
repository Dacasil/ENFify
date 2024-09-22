"""Module with util funcions for data augmentation like making an authentic
audio tampered or chunk larger audios."""

import json
import os
import re
import sys
from glob import glob
from math import ceil
from pathlib import Path

import numpy as np
from locallib import create_auth_tamp_clip
from loguru import logger
from scipy.io import wavfile
from tqdm import tqdm

from enfify import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR

logger.remove()
logger.add(sys.stdout, level="INFO")
np.random.seed(42)


def augmentation(
    dataset_dir: Path,
    interim_dir: Path,
    clip_length: float,
    max_cutlen: float,
    num_clips: int = None,
    max_clips: int = None,
    regex_pattern: str = None,
    overwrite: bool = False,
):
    """Augment a dataset by creating authentic-tampered paired clips from audio files.

    Args:
        dataset_dir (Path): Directory with the original audio files.
        interim_dir (Path): Directory to save the augmented audio files.
        clip_length (float): Length of the clips in seconds.
        max_cutlen (float): Maximum length of the cut in seconds.
        num_clips (int, optional): Number of clips to create from each audio file.
            If None, the number of clips is calculated based on the clip_length.
            Defaults to None.
        max_clips (int, optional): Maximum number of clips to create in total.
            The authentic and tampered version are counted together as one.
            If None, all clips are created. Defaults to None.
        regex_pattern (str, optional): Regular expression pattern to filter files.
            Defaults to None.
        overwrite (bool, optional): Overwrite the interim directory if it already exists.
            Defaults to False.
    """

    if os.path.exists(interim_dir) and not overwrite:
        logger.warning(f"Interim directory {interim_dir} already exists, skipping.")
        return

    if not os.path.exists(dataset_dir):
        logger.error(f"Source directory {dataset_dir} does not exist, skipping.")
        return

    files = sorted(glob(str(dataset_dir / "**" / "*.wav"), recursive=True))
    if regex_pattern:
        regex_pattern = re.compile(regex_pattern)
        files = [f for f in files if regex_pattern.match(os.path.basename(f))]
    logger.info(f"Found {len(files)} files for {os.path.basename(interim_dir)}")

    # PROCESS FILES
    os.makedirs(interim_dir, exist_ok=True)

    cut_info = {}
    i_clip_total = 0
    for file in tqdm(files, desc=f"Augmenting {os.path.basename(interim_dir)}"):
        basename = zero_pad_number_in_filename(os.path.splitext(os.path.basename(file))[0], 2)

        rate, data = wavfile.read(file)

        cliplen_samples = int(clip_length * rate)
        _max_cutlen_samples = int(max_cutlen * rate)

        # Convert stereo to mono by averaging channels
        if data.ndim == 2:
            data = np.mean(data, axis=1)

        # Handle files that are too short
        if len(data) < cliplen_samples:
            logger.warning(f"Skipping {basename} due to insufficient length")
            continue
        elif len(data) < cliplen_samples + _max_cutlen_samples:
            _max_cutlen_samples = len(data) - cliplen_samples
            # TODO: Ugly to use _max_cutlen_samples from now on -> refactor!
            logger.warning(
                f"Decreased max cut length for {basename} to {_max_cutlen_samples / rate:.2f} s due to insufficient length"
            )

        try:
            clips = partition_in_clips(data, cliplen_samples + _max_cutlen_samples, num_clips)
        except ValueError as e:
            logger.warning(f"Skipping {basename} due to {e}")
            continue

        logger.debug(f"Saving {basename} clips")
        for i, clip in enumerate(clips):
            if max_clips and i_clip_total >= max_clips:
                break

            auth_path = interim_dir / f"{basename}-{i:02}-auth.wav"
            tamp_path = interim_dir / f"{basename}-{i:02}-tamp.wav"
            start, cutlen_samples = create_auth_tamp_clip(
                clip,
                rate,
                clip_length,
                _max_cutlen_samples / rate,
                auth_path,
                tamp_path,
            )
            cut_info[tamp_path.name] = {"start": start, "cutlen": cutlen_samples}

            i_clip_total += 1

        if max_clips and i_clip_total >= max_clips:
            break

    cut_info_path = interim_dir / "cut_info.json"
    with open(cut_info_path, "w") as f:
        json.dump(cut_info, f, indent=4)


def partition_in_clips(data, cliplen_samples, num_clips=None):
    """Segment a data array into a number of clips of a given length.
    Used e.g. for augmenting clips of an audio file with a certain length.
    Uses num_clips instead of min_overlap."""

    if len(data) < cliplen_samples:
        raise ValueError("Data length is shorter than clip length")

    if num_clips is None:
        num_clips = ceil(len(data) / cliplen_samples)

    clip_starts = np.linspace(0, len(data) - cliplen_samples, num_clips, dtype=int)
    clips = [data[start : start + cliplen_samples] for start in clip_starts]
    return clips


def zero_pad_number_in_filename(filename, n_digits):
    def replace_match(match):
        number = match.group(0)
        return number.zfill(n_digits)

    new_filename = re.sub(r"\d+", replace_match, filename)

    return new_filename


if __name__ == "__main__":
    augmentation(
        EXTERNAL_DATA_DIR / "Carioca" / "BASE CARIOCA 1",
        INTERIM_DATA_DIR / "Carioca1",
        clip_length=10,
        num_clips=10,
        max_cutlen=2,
        regex_pattern=r"^(HC|MC|HI|MI)\d+\.wav$",
    )
    augmentation(
        EXTERNAL_DATA_DIR / "Carioca" / "BASE CARIOCA 2",
        INTERIM_DATA_DIR / "Carioca2",
        clip_length=10,
        num_clips=10,
        max_cutlen=2,
        regex_pattern=r"^(HF|MF|HC|MC)\d+\.wav$",
    )
    augmentation(
        EXTERNAL_DATA_DIR / "ENF-WHU-Dataset" / "H1",
        INTERIM_DATA_DIR / "WHU",
        clip_length=10,
        num_clips=None,
        max_cutlen=2,
        max_clips=2000,
    )
    augmentation(
        EXTERNAL_DATA_DIR / "ENF-WHU-Dataset" / "H1_ref",
        INTERIM_DATA_DIR / "WHU_ref",
        clip_length=10,
        num_clips=None,
        max_cutlen=2,
        max_clips=2000,
    )
