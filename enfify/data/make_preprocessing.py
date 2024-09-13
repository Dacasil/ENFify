import os
import re
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from loguru import logger
from scipy.io import wavfile
from tqdm import tqdm

from enfify.config import ENFIFY_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from enfify.pipeline import freq_feature_pipeline

logger.remove()
logger.add(sys.stderr, level="ERROR")


def preprocess(dir_path, config):
    glob_pattern = str(Path(dir_path) / "*.wav")
    files = sorted(glob(glob_pattern))

    features_dir = PROCESSED_DATA_DIR / os.path.basename(dir_path)
    os.makedirs(features_dir, exist_ok=True)

    for file in tqdm(files):
        basename = os.path.splitext(os.path.basename(file))[0]
        sample_freq, sig = wavfile.read(file)
        features = freq_feature_pipeline(sig, sample_freq, config)
        np.save(features_dir / f"{basename}.npy", features)


if __name__ == "__main__":
    with open(ENFIFY_DIR / "config_preprocessing.yml", "r") as f:
        config = yaml.safe_load(f)

    preprocess(INTERIM_DATA_DIR / "Carioca1", config)
