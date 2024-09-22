import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import yaml
from loguru import logger
from scipy.io import wavfile
from tqdm import tqdm

from enfify import CONFIG_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, feature_freq_pipeline

logger.remove()
logger.add(sys.stderr, level="ERROR")

DEFAULT_CONFIG_FILE = CONFIG_DIR / "default.yml"
with open(DEFAULT_CONFIG_FILE, "r") as f:
    DEFAULT = yaml.safe_load(f)


def preprocess(interim_path: Path, config: dict):
    if not os.path.exists(interim_path):
        logger.error(f"Interim directory {interim_path} does not exist.")
        return

    features_dir = PROCESSED_DATA_DIR / os.path.basename(interim_path)
    features_dir.mkdir(parents=True)

    files = sorted(glob(str(Path(interim_path) / "*.wav")))

    for file in tqdm(files, desc=f"Processing {os.path.basename(interim_path)}"):
        basename = os.path.splitext(os.path.basename(file))[0]
        outpath = features_dir / f"{basename}.npy"
        if outpath.exists():
            continue
        sample_freq, sig = wavfile.read(file)
        features = feature_freq_pipeline(sig, sample_freq, config)
        np.save(outpath, features)


if __name__ == "__main__":
    interim_dsets = [d for d in INTERIM_DATA_DIR.iterdir() if d.is_dir()]
    for interim_dset in interim_dsets:
        preprocess(interim_dset, DEFAULT)
        # try:
        # except Exception as e:
        #     logger.error(f"Error processing {interim_dset}: {e}")
        #     continue
