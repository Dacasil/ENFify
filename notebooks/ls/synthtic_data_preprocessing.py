import numpy as np
from scipy.io import wavfile
import os

from enfify.preprocessing import downsample_ffmpeg, bandpass_filter
from enfify.enf_enhancement import VMD, RFA

## Audio Paths
UNCUT_PATH = (
    "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/raw/synthetic/10s/uncut/audio"
)
CUT_PATH = "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/raw/synthetic/10s/cut/audio"

## Constants sig, lowcut, highcut, fs, order
DOWNSAMPLE_FS = 1000
BANDPASS_ORDER = 1
BNP_LOW = 49.5
BNP_HIGH = 50.5

# VMD constants
ALPHA = 5000  # Balancing parameter of the data-fidelity constraint
TAU = 0  # Noise-tolerance (no strict fidelity enforcement)
N_MODE = 1  # Number of modes to be recovered
DC = 0  # DC toleration
TOL = 1e-7  # Tolerance of convergence criterion


## Load Audio Signals
uncut_files = [f for f in os.listdir(UNCUT_PATH)].sort()
cut_files = [f for f in os.listdir(CUT_PATH)].sort()

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
down_uncut = []
for i in range():
    down = downsample_ffmpeg(uncut[i][1], uncut[i][0], DOWNSAMPLE_FS)
    down_uncut.append(down)

down_cut = []
for i in range():
    down, down_fs = downsample_ffmpeg(cut[i][1], cut[i][0], DOWNSAMPLE_FS)
    down_cut.append(down)

# Bandpass Filter
band_uncut = []
for i in range(len(down_uncut)):
    band = bandpass_filter(down_uncut[i], BNP_LOW, BNP_HIGH, DOWNSAMPLE_FS, BANDPASS_ORDER)
    band_uncut.append(band)

band_cut = []
for i in range(len(down_uncut)):
    band = bandpass_filter(down_uncut[i], BNP_LOW, BNP_HIGH, DOWNSAMPLE_FS, BANDPASS_ORDER)
    band_uncut.append(band)


# VMD
for i in range(5):
    ALPHA = 5000  # Balancing parameter of the data-fidelity constraint
    TAU = 0  # Noise-tolerance (no strict fidelity enforcement)
    N_MODE = 1  # Number of modes to be recovered
    DC = 0
    TOL = 1e-7  # Tolerance of convergence criterion

    u_clean, _, _ = VMD(band, ALPHA, TAU, N_MODE, DC, TOL)
    vmd_sig = u_clean[0]

# RFA
