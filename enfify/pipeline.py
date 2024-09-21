"""Module that combines steps of the workflow."""

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.signal import get_window

from enfify.enf_enhancement import RFA, RFA_DFT1, RFA_STFT, VMD
from enfify.enf_estimation import (
    STFT,
    freq_estimation_DFT1,
    segmented_freq_estimation_DFT1,
    segmented_phase_estimation_DFT1,
)
from enfify.preprocessing import (  # bandpass_filter,; downsample_ffmpeg,
    bandpass_filter,
    downsample_ffmpeg,
    downsample_scipy,
    downsample_scipy_new,
    extract_spatial_features,
    extract_temporal_features,
    fir_bandpass_filter,
)


def freq_CNN_feature_pipeline(sig, sample_freq, config):
    """
    Processes an audio signal to extract frequency features for a CNN pipeline.

    Args:
        sig (numpy.ndarray): The input audio signal
        sample_freq (float): The sampling frequency of the input signal
        config (dict): Configuration dictionary containing parameters for processing

    Returns:
        numpy.ndarray: The extracted instantaneous frequency features

    Process:
        - Downsamples the signal if enabled
        - Applies a bandpass filter if specified
        - Performs Variational Mode Decomposition (VMD) if enabled
        - Applies a Robust Filtering Algorithm (RFA) if specified
        - Estimates the instantaneous frequencies using DFT1
        - Trims the boundary values from the frequency estimates
    """
    # Nominal ENF
    nom_enf = config["nominal_enf"]

    # Estimate the instantaneous frequency
    freq_estimation_config = config["freq_estimation"]
    n_dft = freq_estimation_config["n_dft"]
    step_size = freq_estimation_config["step_size"]
    window_len = freq_estimation_config["window_len"]

    # Downsampling
    downsample_freq = config["downsampling_frequency_per_nominal_enf"] * config["nominal_enf"]

    if config["downsample_enabled"]:
        sig, sample_freq = downsample_scipy_new(sig, sample_freq, downsample_freq)

    # Bandpass Filter
    bandpass_config = config["bandpass_filter"]
    if bandpass_config["is_enabled"]:
        # lowcut = bandpass_config["lowcut"]
        # highcut = bandpass_config["highcut"]
        # order = bandpass_config["order"]
        sig = fir_bandpass_filter(sig, sample_freq, nom_enf, deltaf=0.6, N=10_000)

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
        # epsilon = RFA_config["epsilon"]

        sig = RFA_STFT(sig, downsample_freq, tau, i, f0, window_len, step_size)

    # Calculate the instantaneous frequencies
    feature_freqs = segmented_freq_estimation_DFT1(
        sig, downsample_freq, n_dft, step_size, window_len
    )

    # Cut the boundary to weaken boundary value problems
    feature_freqs = feature_freqs[40:-40]

    # Plot the frequencies
    if config["plot"]:
        plt.plot(feature_freqs)
        plt.show()

    return feature_freqs


def freq_feature_pipeline(sig, sample_freq, config):
    # Downsampling
    nominal_enf = config["nominal_enf"]
    downsample_freq = config["downsampling_frequency_per_nominal_enf"] * nominal_enf
    sig, sample_freq = downsample_ffmpeg(sig, sample_freq, downsample_freq)

    # Bandpass Filter
    bandpass_config = config["bandpass_filter"]
    lowcut = bandpass_config["lowcut"]
    highcut = bandpass_config["highcut"]
    order = bandpass_config["order"]
    sig = bandpass_filter(sig, lowcut, highcut, sample_freq, order)

    # Frame Splitting and Windowing
    window_type = config["window_type"]
    frame_len_samples = int(config["frame_len"] / 1000 * sample_freq)
    frame_shift = int(frame_len_samples * (1 - config["frame_overlap"]))

    num_frames = (len(sig) - frame_len_samples + frame_shift) // frame_shift
    frames = np.zeros((num_frames, frame_len_samples))

    window = get_window(window_type, frame_len_samples)
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_len_samples
        frames[i] = sig[start:end] * window

    logger.info(f"Number of frames: {num_frames}")
    logger.info(f"Frame length: {frame_len_samples} samples")

    # Estimate the instantaneous frequency
    freq_estimation_config = config["freq_estimation"]
    n_dft = freq_estimation_config["n_dft"]

    feature_freqs = np.array([freq_estimation_DFT1(frame, sample_freq, n_dft) for frame in frames])

    return feature_freqs


def phase_CNNBiLSTM_feature_pipeline(sig, sample_freq, config):
    """
    Processes an audio signal to extract phase features for a CNN BiLSTM pipeline.

    Args:
        sig (numpy.ndarray): The input audio signal.
        sample_freq (float): The sampling frequency of the input signal.
        config (dict): Configuration dictionary containing parameters for processing.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The extracted spatial features.
            - numpy.ndarray: The extracted temporal features.

    Process:
        - Downsamples the signal if enabled.
        - Applies a bandpass filter if enabled.
        - Performs Variational Mode Decomposition (VMD) if enabled.
        - Applies a Robust Filtering Algorithm (RFA) if enabled.
        - Estimates the instantaneous phases using DFT.
        - Trims the boundary values from the phase estimates.
        - Extracts spatial and temporal features for CNN BiLSTM processing.
    """
    # Nominal ENF
    nom_enf = config["nominal_enf"]

    # Estimate the instantaneous phase
    freq_estimation_config = config["phase_estimation"]
    n_dft = freq_estimation_config["n_dft"]
    step_size = freq_estimation_config["step_size"]
    window_len = freq_estimation_config["window_len"]

    # Downsampling
    downsample_freq = config["downsampling_frequency_per_nominal_enf"] * config["nominal_enf"]

    if config["downsample_enabled"]:
        sig, sample_freq = downsample_scipy_new(sig, sample_freq, downsample_freq)

    # Bandpass Filter
    bandpass_config = config["bandpass_filter"]
    if bandpass_config["is_enabled"]:
        # lowcut = bandpass_config["lowcut"]
        # highcut = bandpass_config["highcut"]
        # order = bandpass_config["order"]
        sig = fir_bandpass_filter(sig, sample_freq, nom_enf, deltaf=1, N=1_000)

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
        # epsilon = RFA_config["epsilon"]

        sig = RFA_STFT(sig, downsample_freq, tau, i, f0, window_len, step_size)

    # Calculate the instantaneous phases
    feature_phases = segmented_phase_estimation_DFT1(
        sig, downsample_freq, nom_enf, n_dft, step_size, window_len
    )

    # Cut the boundary to weaken boundary value problems
    feature_phases = feature_phases[40:-40]

    # Plot the frequencies
    if config["plot"]:
        plt.plot(feature_phases)
        plt.show()

    # CNN BiLSTM feature processing
    feature_config = config["feature_matrices"]
    sn = feature_config["sn"]
    fl = feature_config["fl"]
    fn = feature_config["fn"]

    spatial_features = extract_spatial_features(feature_phases, sn)
    temporal_features = extract_temporal_features(feature_phases, fl, fn)

    return spatial_features, temporal_features


def freq_CNNBiLSTM_feature_pipeline(sig, sample_freq, config):
    """
    Processes an audio signal to extract frequency features for a CNN BiLSTM pipeline.

    Args:
        sig (numpy.ndarray): The input audio signal
        sample_freq (float): The sampling frequency of the input signal
        config (dict): Configuration dictionary containing parameters for processing

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The extracted spatial features.
            - numpy.ndarray: The extracted temporal features.

    Process:
        - Downsamples the signal if enabled.
        - Applies a bandpass filter if spenabled.
        - Performs Variational Mode Decomposition (VMD) if enabled.
        - Applies a Robust Filtering Algorithm (RFA) if enabled.
        - Estimates the instantaneous frequencies using STFT.
        - Trims the boundary values from the frequency estimates.
        - Extracts spatial and temporal features for CNN BiLSTM processing.
    """
    # Nominal ENF
    nom_enf = config["nominal_enf"]

    # Estimate the instantaneous phase
    freq_estimation_config = config["phase_estimation"]
    step_size = freq_estimation_config["step_size"]
    window_len = freq_estimation_config["window_len"]
    n_dft = freq_estimation_config["n_dft"]

    # Downsampling
    downsample_freq = config["downsampling_frequency_per_nominal_enf"] * config["nominal_enf"]

    if config["downsample_enabled"]:
        sig, sample_freq = downsample_scipy_new(sig, sample_freq, downsample_freq)

    # Bandpass Filter
    bandpass_config = config["bandpass_filter"]
    if bandpass_config["is_enabled"]:
        # lowcut = bandpass_config["lowcut"]
        # highcut = bandpass_config["highcut"]
        # order = bandpass_config["order"]
        sig = fir_bandpass_filter(sig, sample_freq, nom_enf, deltaf=0.6, N=10_000)

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
        # epsilon = RFA_config["epsilon"]
        sig = RFA_DFT1(sig, downsample_freq, tau, i, f0, window_len, step_size, n_dft)

    # Calculate the instantaneous frequencies
    feature_freqs = STFT(sig, downsample_freq, step_size, window_len)

    # Cut the boundary to weaken boundary value problems
    feature_freqs = feature_freqs[40:-40]

    # Plot the frequencies
    if config["plot"]:
        plt.plot(feature_freqs)
        plt.show()

    # CNN BiLSTM feature processing
    feature_config = config["feature_matrices"]
    sn = feature_config["sn"]
    fl = feature_config["fl"]
    fn = feature_config["fn"]

    spatial_features = extract_spatial_features(feature_freqs, sn)
    temporal_features = extract_temporal_features(feature_freqs, fl, fn)

    return spatial_features, temporal_features
