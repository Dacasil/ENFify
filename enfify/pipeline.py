"""Module that combines steps of the workflow."""

import numpy as np
from scipy.signal import get_window

from enfify.preprocessing import bandpass_filter, downsample_scipy
from enfify.enf_enhancement import VMD, RFA
from enfify.enf_estimation import segmented_freq_estimation_DFT1


def frame_split(sig, window_type, frame_len, frame_shift):
    """Split the signal into frames."""
    num_frames = (len(sig) - frame_len + frame_shift) // frame_shift
    frames = np.zeros((num_frames, frame_len))

    window = get_window(window_type, frame_len)
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_len
        frames[i] = sig[start:end] * window

    return frames


def freq_feature_pipeline(sig, sample_freq, config):
    # Downsampling
    downsample_freq = config["downsampling_frequency_per_nominal_enf"] * config["nominal_enf"]
    sig, sample_freq = downsample_scipy(sig, sample_freq, downsample_freq)

    # Frame Splitting and Windowing
    window_type = config["window_type"]
    frame_len = int(config["frame_len"] / 1000 * sample_freq)
    frame_shift = int(frame_len * config["frame_overlap"])

    frames = frame_split(sig, window_type, frame_len, frame_shift)

    # Bandpass Filter
    bandpass_config = config["bandpass_filter"]
    lowcut = bandpass_config["lowcut"]
    highcut = bandpass_config["highcut"]
    order = bandpass_config["order"]
    frames = bandpass_filter(frames, lowcut, highcut, sample_freq, order)

    # frames = VMD(frames, sample_freq, config)

    # Robust Filtering Algorithm
    # frames = RFA(frames, sample_freq, config)

    # STFT
    _amplitudes = np.abs(np.fft.rfft(frames, axis=-1))
    _frequencies = np.fft.rfftfreq(frame_len, d=1 / sample_freq)
    feature_freqs = _frequencies[np.argmax(_amplitudes, axis=-1)]

    return feature_freqs


def freq_ls_feature_pipeline(sig, sample_freq, config):

    # Nominal ENF
    nom_enf = config["nominal_enf"]

    # Downsampling
    downsample_freq = config["downsampling_frequency_per_nominal_enf"] * config["nominal_enf"]

    if config["downsample_enabled"]:
        sig, sample_freq = downsample_scipy(sig, sample_freq, downsample_freq)

    # Bandpass Filter
    bandpass_config = config["bandpass_filter"]
    if bandpass_config["is_enabled"]:
        lowcut = bandpass_config["lowcut"]
        highcut = bandpass_config["highcut"]
        order = bandpass_config["order"]
        sig = bandpass_filter(sig, lowcut, highcut, sample_freq, order)

    # Variational Mode Decomposition
    VMD_config = config["VMD"]
    if VMD_config["is_enabled"]:
        loop = VMD_config["loop"]
        alpha = VMD_config["alpha"]
        tau = VMD_config["tau"]
        n_mode = VMD_config["n_mode"]
        DC = VMD_config["DC"]
        tol = VMD_config["tol"]

        for i in range(loop):
            u_clean, _, _ = VMD(sig, alpha, tau, n_mode, DC, tol)
            sig = u_clean[0]

    # Robust Filtering Algorithm
    RFA_config = config["RFA"]
    if RFA_config["is_enabled"]:
        f0 = RFA_config["f0"]
        i = RFA_config["I"]
        tau = RFA_config["tau"]
        epsilon = RFA_config["epsilon"]

        sig = RFA(sig, downsample_freq, tau, epsilon, i, f0)

    # Estimate the instantaneous frequency
    freq_estimation_config = config["freq_estimation"]
    n_dft = freq_estimation_config["n_dft"]
    num_cycles = freq_estimation_config["num_cycles"]

    feature_freqs = segmented_freq_estimation_DFT1(
        sig, downsample_freq, num_cycles, n_dft, nom_enf
    )

    return feature_freqs
