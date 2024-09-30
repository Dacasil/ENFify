"""Module for preprocessing the ENF signal."""

import os
import tempfile
import math
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from loguru import logger
from scipy.io import wavfile
from scipy.signal import decimate, resample, butter, lfilter


# .......................Downsampling........................#


# The prefered Downsample function
def downsample_scipy_new(sig, sampling_rate, downsampling_rate):
    """Apply downsampling using scipy and remove DC components

    Args:
        sig (numpy.ndarray): Audio signal
        sampling_rate (float): Current downsample sampling frequency
        downsampling_rate (float): Desired downsample sampling frequency

    Returns:
        numpy.ndarray: Downsampled signal
        float: Downsample sampling frequency
    """
    # Remove DC components
    dc_sig = sig - np.mean(sig)

    # Resample the audio signal
    resampled_sig = resample(dc_sig, int(len(dc_sig) * downsampling_rate / sampling_rate))

    # Add here Librosa instead of resample maybe

    return resampled_sig, downsampling_rate


def lowpass_filter(signal, cutoff, sampling_rate, order=5):
    """Apply a Butterworth lowpass filter for antialiasing.

    Args:
        signal (numpy.ndarray): Audio signal
        cutoff (float): Cutoff Frequency of the filter (in Hz)
        sampling_rate (float): Sampling frequency of the signal
        order (int): Order of the filter

    Returns:
        numpy.ndarray: Filtered audio signal
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return lfilter(b, a, signal)


def downsample_scipy(sig, sampling_rate, downsampling_rate):
    """Downsample a numpy array using scipy, with antialiasing.

    Args:
        sig (numpy.ndarray): Audio signal
        sampling_rate (int or float): Current sampling frequency
        downsampling_rate (int or float): Desired sampling frequency

    Returns:
        numpy.ndarray: Downsampled audio signal
        float: Downsampling frequency
    """

    if downsampling_rate <= 0:
        raise ValueError("Target sampling rate must be greater than 0.")

    if downsampling_rate >= sampling_rate:
        logger.warning(
            "Not downsampling since the target sampling rate is greater than the current sampling rate."
        )
        return sig, sampling_rate

    if sampling_rate % downsampling_rate == 0:
        # If the target sampling rate is an integer multiple of the original rate
        decimation_factor = int(sampling_rate // downsampling_rate)
        return (
            decimate(sig, decimation_factor),
            downsampling_rate,
        )  # Antialiasing is integrated here
    else:
        # Otherwise, use resampling
        # Log a warning about the need for an antialiasing filter
        logger.warning(
            f"The target sampling rate ({downsample_rate}) is not an integer multiple of the current sampling rate ({sample_rate}). Resampling is used, which does not have an integrated antialiasing filter. Applying a lowpass filter."
        )

        # Apply a lowpass filter to ensure antialiasing
        cutoff_frequency = downsampling_rate / 2.0
        filtered_sig = lowpass_filter(sig, cutoff_frequency, sampling_rate)
        num_samples = int(len(filtered_sig) * downsampling_rate / sampling_rate)
        return resample(filtered_sig, num_samples), downsampling_rate


# ......................Bandpass Filter......................#
def butter_bandpass_test(lowcut, highcut, sampling_rate, order=5):
    """Test the butter bandpass filter

    Args:
        lowcut (int or float): Lower limit of the wanted frequency
        highcut (int or float): Upper limit of the wanted frequency
        sampling_rate (int or float): Sampling frequency
        order (int or float): Precision order of the bandpass filter

    Returns:
        Numpy Array: Bandpass Filter
    """

    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype="band", output="sos")
    w, h = signal.sosfreqz(sos, worN=20000)
    plt.semilogx((sampling_rate * 0.5 / np.pi) * w, abs(h))
    return sos


# ................. Spatial and temporal features for the  CNN-BiLSTM Network ..................#


def extract_temporal_features(psi_1_phases, fl, fn):
    """
    Extracts temporal features from phase sequence features.

    Args:
        psi_1_list (list of numpy.ndarray): A list of phase sequence features for multiple audio files
        fl (int): The number of phase points contained in each frame (frame length)

    Returns:
        list of numpy.ndarray: A list of temporal feature matrices for each audio file
    """

    current_len = len(psi_1_phases)

    overlap = fl - math.floor(current_len / fn)

    # Split the phase sequence into frames using the calculated overlap
    frames = []
    for i in range(0, current_len - fl + 1, overlap):
        frame = psi_1_phases[i : i + fl]
        frames.append(frame)

    # Cases where the last frame is smaller than `fl`
    if len(psi_1_phases) % fl != 0:
        frame = psi_1_phases[-fl:]
        frames.append(frame)

    # Reshape the frames into a temporal feature matrix T F fl × fn
    feature_matrix = np.zeros((fl, fn))

    # Matrix
    for i in range(min(fn, len(frames))):
        feature_matrix[:, i] = frames[i]

    return feature_matrix


def extract_spatial_features(psi_1_phases, sn):
    """
    Extracts spatial features from phase sequence features.

    Args:
        psi_1_list (numpy.ndarray[float]): A list of phase sequence features for multiple audio files

    Returns:
        list of numpy.ndarray: A list of spatial feature matrices for each audio file
    """

    ML = sn**2

    current_len = len(psi_1_phases)

    overlap = sn - math.ceil((ML - sn) / (current_len - sn))

    # Split the frame
    num_frames = (current_len - sn) // overlap + 1
    frames = []

    for i in range(0, num_frames * overlap, overlap):
        if i + sn <= current_len:
            frame = psi_1_phases[i : i + sn]
            frames.append(frame)
        else:
            break

    # Reshape into a spatial feature matrix (S Fsn×sn)
    feature_matrix = np.zeros((sn, sn))

    for i in range(min(sn, len(frames))):
        feature_matrix[i, : len(frames[i])] = frames[i]

    return feature_matrix


# .................Cut signal..................#


def cut_signal(sig, cut_start, cut_len):
    """Cuts numpy arrays

    Args:
        sig (nparray): numpy array signal
        F_DS (int or float): Sampling frequency of the signal

    Returns:
        _type_: Cut numpy array
    """
    cut_end = cut_start + cut_len
    cut_sig = np.concatenate((sig[:cut_start], sig[cut_end:]))

    return cut_sig


def cut_out_signal(sig, cut_start, cut_len):
    """Cuts out numpy arrays

    Args:
        sig (nparray): numpy array signal
        F_DS (int or float): Sampling frequency of the signal

    Returns:
        _type_: Cut numpy array
    """

    cut_end = cut_start + cut_len
    cut_sig = sig[cut_start : cut_end + 1]

    return cut_sig
