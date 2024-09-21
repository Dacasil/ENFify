import numpy as np
from scipy.io import wavfile
import os
from tqdm import tqdm

from enfify.preprocessing import (
    downsample_ffmpeg,
    bandpass_filter,
    extract_number,
    cut_extract_number,
)
from enfify.enf_estimation import segmented_freq_estimation_DFT1

## Audio Paths
UNCUT_PATH = (
    "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/raw/synthetic/10s/uncut/audio"
)
CUT_PATH = "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/raw/synthetic/10s/cut/audio"

PATH = "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/processed/Synthetic_Data/10s"

## Constants sig, lowcut, highcut, fs, order
DOWNSAMPLE_FS = 1000
BANDPASS_ORDER = 1
NOM_ENF = 50
BNP_LOW = 49.5
BNP_HIGH = 50.5

# Frequency estimation
NUM_CYCLES = 10
N_DFT = 20_000

## Load Audio Signals
uncut_files = sorted(os.listdir(UNCUT_PATH))
uncut_files.sort(key=extract_number)

cut_files = sorted(os.listdir(CUT_PATH))
cut_files.sort(key=cut_extract_number)

uncut = []
for wav in uncut_files:
    pth = os.path.join(UNCUT_PATH, wav)
    sample_rate, data = wavfile.read(pth)
    uncut.append([sample_rate, data])

cut = []
for wav in cut_files:
    pth = os.path.join(CUT_PATH, wav)
    sample_rate, data = wavfile.read(pth)
    cut.append([sample_rate, data])

# Downsampling
down_uncut = [downsample_ffmpeg(data[1], data[0], DOWNSAMPLE_FS)[0] for data in tqdm(uncut)]
down_cut = [downsample_ffmpeg(data[1], data[0], DOWNSAMPLE_FS)[0] for data in tqdm(cut)]

# Bandpass Filter
band_uncut = [
    bandpass_filter(down, BNP_LOW, BNP_HIGH, DOWNSAMPLE_FS, BANDPASS_ORDER)
    for down in tqdm(down_uncut)
]
band_cut = [
    bandpass_filter(down, BNP_LOW, BNP_HIGH, DOWNSAMPLE_FS, BANDPASS_ORDER)
    for down in tqdm(down_cut)
]

# Calculate Frequencies
uncut_freqs = [
    segmented_freq_estimation_DFT1(band_sig, DOWNSAMPLE_FS, NUM_CYCLES, N_DFT, NOM_ENF)
    for band_sig in tqdm(band_uncut)
]
cut_freqs = [
    segmented_freq_estimation_DFT1(band_sig, DOWNSAMPLE_FS, NUM_CYCLES, N_DFT, NOM_ENF)
    for band_sig in tqdm(band_cut)
]

# Label Combine and shuffle the Data
labels = np.concatenate([np.ones(len(cut_freqs)), np.zeros(len(uncut_freqs))])
freq_data = cut_freqs + uncut_freqs
indices = np.random.permutation(len(freq_data))

freqs_data = np.array([freq_data[i] for i in indices], dtype=object)
labels = labels[indices]

# Save frequencies and labels
np.save(PATH + "/" + "synth_freqs_10s.npy", freqs_data)
np.save(PATH + "/" + "synth_freqs_10s_labels.npy", labels)
