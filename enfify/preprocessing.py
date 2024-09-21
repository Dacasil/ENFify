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


def downsample_ffmpeg(sig, sampling_rate, downsampling_rate):
    """
    Downsamples an audio signal using FFmpeg

    Args:
        sig (numpy.ndarray): Audio signal
        sampling_rate (float): The current sampling rate of the input signal
        downsampling_rate (float): The desired downsampling rate

    Returns:
        numpy.ndarray: Downsampled audio signal
        float: Downsampling frequency

    """
    with (
        tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as input_file,
        tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as output_file,
    ):
        input_path = input_file.name
        output_path = output_file.name
        wavfile.write(input_path, sampling_rate, sig)

        ffmpeg.input(input_path).output(
            output_path, ar=downsampling_rate
        ).overwrite_output().global_args("-loglevel", "error").run()

        downsampling_rate, downsampled_sig = wavfile.read(output_path)

        os.remove(input_path)
        os.remove(output_path)

    return downsampled_sig, downsampling_rate


# ......................Bandpass Filter......................#


# The prefered bandpassfilter
def fir_bandpass_filter(sig, sampling_rate, nominal_enf, deltaf, N):
    """
    Applies a zero-phase FIR bandpass filter to the input signal.

    Args:
        sig (numpy.ndarray): Audio signal
        sampling_rate (float): Sampling frequency
        nominal_enf (float, optional): Center frequency of the bandpass filter (default is 50 Hz)
        deltaf (float, optional): Bandwidth of the filter in Hz (default is 0.6 Hz)
        N (int, optional): Order (number of taps) of the FIR filter

    Returns:
        numpy.ndarray: Filtered audio signal
    """

    # Filter parameters
    f1 = nominal_enf - deltaf / 2
    f2 = nominal_enf + deltaf / 2

    # Normalize the frequencies to [0, 1] (Nyquist is 1)
    w1 = f1 * 2 * np.pi / sampling_rate
    w2 = f2 * 2 * np.pi / sampling_rate
    Wn = [w1 / np.pi, w2 / np.pi]
    # Wn = [f1, f2]
    # Wn = [w1, w2]

    # h = signal.firwin(N, Wn, pass_zero=False, fs=sampling_rate)  # FIR Filter Design
    h = signal.firwin(
        N, Wn, width=0.0012, window="hann", pass_zero=False, scale=True, fs=sampling_rate
    )  # FIR Filter Design
    sig_len = len(sig)  # Length of the signal

    # Filter the signal with zero-padding for shorter signals
    if sig_len <= 3 * N:
        pad_length = (
            int(1.5 * N - sig_len // 2) + 50
        )  # Pad the signal with zeros to avoid edge effects
        padded_signal = np.concatenate((np.zeros(pad_length), sig, np.zeros(pad_length)))

        # Apply the zero-phase filtering
        filtered_signal = signal.filtfilt(h, 1, padded_signal)

        # Remove the padding
        sig = sig[pad_length : pad_length + sig_len]
        filtered_signal = filtered_signal[pad_length : pad_length + sig_len]

    # For longer signals, a fixed padding of 1000 samples is used
    else:
        padded_signal = np.concatenate((np.zeros(1000), sig, np.zeros(1000)))

        # Apply the zero-phase filtering
        filtered_signal = signal.filtfilt(h, 1, padded_signal)

        # Remove the padding
        sig = sig[1000 : 1000 + sig_len]
        filtered_signal = filtered_signal[1000 : 1000 + sig_len]

    """
    w, h = signal.freqz(h, worN=8000, fs=sampling_rate)
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.title("Frequency response of the FIR filter")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.grid()
    plt.show()"""

    return filtered_signal


def bandpass_filter(sig, lowcut, highcut, sampling_rate, order):
    """Bandpass Filter to cut out unwanted frequencys of a signal using Butterworth

    Args:
        sig (numpy array_):Audio Signal
        lowcut (int or float): Lower limit of the wanted frequency (Hz)
        highcut (int or float): Upper limit of the wanted frequency (Hz)
        sampling_rate (int or float): Sampling frequency
        order (int or float): Order of the bandpass filter

    Returns:
        Numpy Array: Bandpassed signal
    """

    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype="band", output="sos")
    bandpass_sig = signal.sosfiltfilt(sos, sig)

    return bandpass_sig


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
