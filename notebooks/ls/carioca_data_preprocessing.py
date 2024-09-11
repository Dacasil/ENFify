import numpy as np
import os
from glob import glob
import yaml
from scipy.io import wavfile
from tqdm import tqdm

from enfify.config import ENFIFY_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from enfify.pipeline import freq_feature_pipeline

# Paths
FEATURE_PATH = PROCESSED_DATA_DIR / "Carioca1_new"
os.makedirs(FEATURE_PATH, exist_ok=True)

# Uncut
uncut_files = str(INTERIM_DATA_DIR / "Carioca1" / "authentic" / "*.wav")
uncut_files = sorted(glob(uncut_files))

for file in tqdm(uncut_files):
    basename = os.path.splitext(os.path.basename(file))[0]
    sample_freq, sig = wavfile.read(file)
    with open(ENFIFY_DIR / "config_nature.yml", "r") as f:
        config = yaml.safe_load(f)
    uncut_freqs = freq_feature_pipeline(sig, sample_freq, config)

# Cut
cut_files = str(INTERIM_DATA_DIR / "Carioca1" / "tampered" / "*.wav")
cut_files = sorted(glob(cut_files))

for file in tqdm(cut_files):
    basename = os.path.splitext(os.path.basename(file))[0]
    sample_freq, sig = wavfile.read(file)
    with open(ENFIFY_DIR / "config_nature.yml", "r") as f:
        config = yaml.safe_load(f)
    cut_freqs = freq_feature_pipeline(sig, sample_freq, config)

# Label Combine and shuffle the Data
labels = np.concatenate([np.ones(len(cut_freqs)), np.zeros(len(uncut_freqs))])
freq_data = cut_freqs + uncut_freqs
indices = np.random.permutation(len(freq_data))

freqs_data = np.array([freq_data[i] for i in indices], dtype=object)
labels = labels[indices]

# Save the data
np.save(FEATURE_PATH + "/" + "carioca_freqs_10s.npy", freqs_data)
np.save(FEATURE_PATH + "/" + "carioca_freqs_10s_labels.npy", labels)
