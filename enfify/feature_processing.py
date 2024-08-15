import os
from glob import glob
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import yaml
from analyses import classification_dacasil
from enf_enhancement import VariationalModeDecomposition
from enf_estimation import segmented_phase_estimation_DFT0, segmented_phase_estimation_hilbert
from preprocessing import bandpass_filter, downsampling_alpha
from Rodriguez_Audio_Authenticity import feature
from scipy import stats
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import read_wavfile, add_defaults


def get_hilbert_phase(sig, fs, config=None):
    # Config
    # MayDo: Erst defaults als config variable laden, dann mit config.yml updaten. Extra funktion nötig weil nested dicts.
    if config is None:
        config = {}
    with open(os.path.dirname(__file__) + "/defaults.yml", "r") as f:
        defaults = yaml.safe_load(f)
    add_defaults(config, defaults)

    # ENF EXTRACTION

    # Downsampling
    downsample_config = config["downsample"]
    if downsample_config["is_enabled"]:
        f_ds = downsample_config["downsampling_frequency"]

        sig, fs = downsampling_alpha(sig, fs, f_ds)

    # Bandpass Filter
    bandpass_config = config["bandpassfilter"]
    if bandpass_config["is_enabled"]:
        lowcut = bandpass_config["lowcut"]
        highcut = bandpass_config["highcut"]
        sig = bandpass_filter(sig, lowcut, highcut, fs, 1)

    # Variational Mode Decomposition
    vmd_config = config["VMD"]
    if vmd_config["is_enabled"]:
        alpha = vmd_config["alpha"]
        tau = vmd_config["tau"]
        n_mode = vmd_config["n_mode"]
        DC = vmd_config["DC"]
        tol = vmd_config["tol"]

        u_clean, _, _ = VariationalModeDecomposition(sig, alpha, tau, n_mode, DC, tol)
        sig = u_clean[0]

    # ENF ANALYSIS

    # Phase Estimation
    nom_enf = config["nominal_enf"]

    phase_config = config["phase_estimation"]
    num_cycles = phase_config["num_cycles"]
    # n_dft = phase_config["n_dft"]

    time = len(sig) / fs

    # # hilbert phase estimation
    phase = segmented_phase_estimation_hilbert(sig, fs, num_cycles, nom_enf)
    x = np.linspace(0.0, time, len(phase))

    # DFT0 instantaneous phase estimation
    # phases = segmented_phase_estimation_DFT0(sig, fs, num_cycles, n_dft, nom_enf)
    # x = np.linspace(0.0, time, len(phases))
    return x, phase
