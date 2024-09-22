import concurrent.futures
import shutil
import sys
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


def preprocess(interim_path: Path, config: dict, overwrite: bool = True):  # !!!
    if not interim_path.exists():
        logger.error(f"Interim directory {interim_path} does not exist.")
        return

    features_dir = PROCESSED_DATA_DIR / interim_path.name

    if features_dir.exists():
        if not overwrite:
            logger.warning(f"Features directory {features_dir} already exists, skipping.")
            return
        logger.warning(f"Overwriting existing features directory {features_dir}")
        shutil.rmtree(features_dir)

    features_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(interim_path.glob("*.wav"))

    for file in tqdm(files, desc=f"Processing {interim_path.name}"):
        # for file in files:
        basename = file.stem
        outpath = features_dir / f"{basename}.npy"
        sample_freq, sig = wavfile.read(file)
        features = feature_freq_pipeline(sig, sample_freq, config)
        np.save(outpath, features)


# if __name__ == "__main__":
#     with open(CONFIG_DIR / "config_carioca.yml", "r") as f:
#         config_carioca = yaml.safe_load(f)

#     preprocess(INTERIM_DATA_DIR / "Carioca1", config_carioca)
#     preprocess(INTERIM_DATA_DIR / "Carioca2", config_carioca)
#     preprocess(INTERIM_DATA_DIR / "Synthetic", DEFAULT)
#     preprocess(INTERIM_DATA_DIR / "WHU", DEFAULT)
#     preprocess(INTERIM_DATA_DIR / "WHU_ref", DEFAULT)

if __name__ == "__main__":
    with open(CONFIG_DIR / "config_carioca.yml", "r") as f:
        config_carioca = yaml.safe_load(f)

    paths = [
        (INTERIM_DATA_DIR / "Carioca1", config_carioca),
        (INTERIM_DATA_DIR / "Carioca2", config_carioca),
        (INTERIM_DATA_DIR / "Synthetic", DEFAULT),
        (INTERIM_DATA_DIR / "WHU", DEFAULT),
        (INTERIM_DATA_DIR / "WHU_ref", DEFAULT),
    ]

    # Parallelverarbeitung der Datasets
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(preprocess, path, config): path for path, config in paths}
