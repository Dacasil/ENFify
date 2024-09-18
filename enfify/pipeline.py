"""Module that combines steps of the workflow."""

import matplotlib.pyplot as plt

from enfify.preprocessing import (
    bandpass_filter,
    downsample_ffmpeg,
    extract_spatial_features,
    extract_temporal_features,
)
from enfify.enf_enhancement import VMD, RFA
from enfify.enf_estimation import (
    segmented_freq_estimation_DFT1,
    segmented_phase_estimation_DFT1,
    STFT,
)


def freq_CNN_feature_pipeline(sig, sample_freq, config):

    # Downsampling
    downsample_freq = config["downsampling_frequency_per_nominal_enf"] * config["nominal_enf"]

    if config["downsample_enabled"]:
        sig, sample_freq = downsample_ffmpeg(sig, sample_freq, downsample_freq)

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
    step_size = freq_estimation_config["step_size"]
    window_len = freq_estimation_config["window_len"]

    feature_freqs = segmented_freq_estimation_DFT1(
        sig, downsample_freq, n_dft, step_size, window_len
    )

    return feature_freqs


def phase_CNNBiLSTM_feature_pipeline(sig, sample_freq, config):

    # Nominal ENF
    nom_enf = config["nominal_enf"]

    # Downsampling
    downsample_freq = config["downsampling_frequency_per_nominal_enf"] * config["nominal_enf"]

    if config["downsample_enabled"]:
        sig, sample_freq = downsample_ffmpeg(sig, sample_freq, downsample_freq)

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

    # Estimate the instantaneous phase
    freq_estimation_config = config["phase_estimation"]
    n_dft = freq_estimation_config["n_dft"]
    step_size = freq_estimation_config["step_size"]
    window_len = freq_estimation_config["window_len"]

    feature_phases = segmented_phase_estimation_DFT1(
        sig, downsample_freq, nom_enf, n_dft, step_size, window_len
    )

    # Cut the boundary to weaken boundary value problems
    feature_phases = feature_phases[40:-40]

    if config["plot"]:
        plt.plot(feature_phases)
        plt.show()

    feature_config = config["feature_matrices"]
    sn = feature_config["sn"]
    fl = feature_config["fl"]
    fn = feature_config["fn"]

    spatial_features = extract_spatial_features(feature_phases, sn)
    temporal_features = extract_temporal_features(feature_phases, fl, fn)

    return spatial_features, temporal_features


def freq_CNNBiLSTM_feature_pipeline(sig, sample_freq, config):

    # Nominal ENF
    # nom_enf = config["nominal_enf"]

    # Downsampling
    downsample_freq = config["downsampling_frequency_per_nominal_enf"] * config["nominal_enf"]

    if config["downsample_enabled"]:
        sig, sample_freq = downsample_ffmpeg(sig, sample_freq, downsample_freq)

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

    # Estimate the instantaneous phase
    freq_estimation_config = config["phase_estimation"]
    step_size = freq_estimation_config["step_size"]
    window_len = freq_estimation_config["window_len"]

    feature_freqs = STFT(sig, downsample_freq, step_size, window_len)

    feature_freqs = feature_freqs[40:-40]

    if config["plot"]:
        plt.plot(feature_freqs)
        plt.show()

    feature_config = config["feature_matrices"]
    sn = feature_config["sn"]
    fl = feature_config["fl"]
    fn = feature_config["fn"]

    spatial_features = extract_spatial_features(feature_freqs, sn)
    temporal_features = extract_temporal_features(feature_freqs, fl, fn)

    return spatial_features, temporal_features
