import numpy as np
import os
from glob import glob
import yaml
from scipy.io import wavfile
from tqdm import tqdm
from pathlib import Path
from enfify.config import ENFIFY_DIR
from enfify.pipeline import freq_CNNBiLSTM_feature_pipeline

RAW_DATA_DIR = Path("/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/interim")
PROCESSED_DATA_DIR = Path("/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/processed")

# Paths
FEATURE_PATH = Path(PROCESSED_DATA_DIR / "Carioca_10s_freqs_CNNBiLSTM")
os.makedirs(FEATURE_PATH, exist_ok=True)


# Uncut
uncut_files = Path(RAW_DATA_DIR / "Carioca1" / "authentic" / "*.wav")
uncut_files = sorted(glob(str(uncut_files)))
uncut_spatial = []
uncut_temporal = []

# Cut
cut_files = Path(RAW_DATA_DIR / "Carioca1" / "tampered" / "*.wav")
cut_files = sorted(glob(str(cut_files)))
cut_spatial = []
cut_temporal = []


for file in tqdm(cut_files):
    basename = os.path.splitext(os.path.basename(file))[0]
    sample_freq, sig = wavfile.read(file)
    with open(ENFIFY_DIR / "config_springer.yml", "r") as f:
        config = yaml.safe_load(f)
    cut_spatial_phase, cut_temporal_phase = freq_CNNBiLSTM_feature_pipeline(
        sig, sample_freq, config
    )
    cut_spatial.append(cut_spatial_phase)
    cut_temporal.append(cut_temporal_phase)


for file in tqdm(uncut_files):
    basename = os.path.splitext(os.path.basename(file))[0]
    sample_freq, sig = wavfile.read(file)
    with open(ENFIFY_DIR / "config_springer.yml", "r") as f:
        config = yaml.safe_load(f)
    uncut_spatial_phase, uncut_temporal_phase = freq_CNNBiLSTM_feature_pipeline(
        sig, sample_freq, config
    )
    uncut_spatial.append(uncut_spatial_phase)
    uncut_temporal.append(uncut_temporal_phase)


# Label Combine and shuffle the Data
labels = np.concatenate([np.ones(len(cut_spatial)), np.zeros(len(cut_temporal))])
spatial_data = uncut_spatial + cut_spatial
temporal_data = uncut_temporal + cut_temporal
indices = np.random.permutation(len(spatial_data))

print(f"Spatial Data {len(spatial_data)}")
print(f"Temporal Data {len(temporal_data)}")

spatial_data = np.array([spatial_data[i] for i in indices], dtype=object)
temporal_data = np.array([temporal_data[i] for i in indices], dtype=object)
labels = labels[indices]

# Save the data
np.save(str(FEATURE_PATH) + "/" + "synth_temporal_freqs_40s.npy", temporal_data)
np.save(str(FEATURE_PATH) + "/" + "synth_spatial_freqs_40s.npy", spatial_data)
np.save(str(FEATURE_PATH) + "/" + "synth_labels_40s_freqs.npy", labels)
