import os
import tempfile
from math import ceil, floor

import ffmpeg
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.signal import resample


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


# The prefered bandpassfilter
def fir_bandpass_filter(sig, sampling_rate, lowcut, highcut, order):
    """
    Applies a zero-phase FIR bandpass filter to the input signal.

    Args:
        sig (numpy.ndarray): Audio signal
        sampling_rate (float): Sampling frequency
        lowcut (float): Lower limit of the wanted frequency (Hz)
        highcut (float): Upper limit of the wanted frequency (Hz)
        order (int): Order (number of taps) of the FIR filter

    Returns:
        numpy.ndarray: Filtered audio signal
    """

    # Normalize the frequencies to [0, 1] (Nyquist is 1)
    w1 = lowcut * 2 * np.pi / sampling_rate
    w2 = highcut * 2 * np.pi / sampling_rate
    Wn = [w1 / np.pi, w2 / np.pi]

    h = signal.firwin(order, Wn, pass_zero="bandpass")  # FIR Filter Design
    sig_len = len(sig)  # Length of the signal

    # Filter the signal with zero-padding for shorter signals
    if sig_len <= 3 * order:
        pad_length = (
            int(1.5 * order - sig_len // 2) + 50
        )  # Pad the signal with zeros to avoid edge effects
        padded_signal = np.concatenate((np.zeros(pad_length), sig, np.zeros(pad_length)))

        # Apply the zero-phase filtering
        filtered_signal = signal.filtfilt(
            h, 1, padded_signal, axis=0, padtype="odd", padlen=3 * (max(len(h), 1) - 1)
        )

        # Remove the padding
        sig = sig[pad_length : pad_length + sig_len]
        filtered_signal = filtered_signal[pad_length : pad_length + sig_len]

    # For longer signals, a fixed padding of 1000 samples is used
    else:
        padded_signal = np.concatenate((np.zeros(1000), sig, np.zeros(1000)))

        # Apply the zero-phase filtering
        filtered_signal = signal.filtfilt(
            h, 1, padded_signal, padtype="odd", padlen=3 * (max(len(h), 1) - 1)
        )

        # Remove the padding
        sig = sig[1000 : 1000 + sig_len]
        filtered_signal = filtered_signal[1000 : 1000 + sig_len]

    return filtered_signal


def butterworth_bandpass_filter(sig, sampling_rate, lowcut, highcut, order):
    """Bandpass Filter to cut out unwanted frequencys of a signal using Butterworth

    Args:
        sig (numpy.array):Audio Signal
        lowcut (float): Lower limit of the wanted frequency (Hz)
        highcut (float): Upper limit of the wanted frequency (Hz)
        sampling_rate (loat): Sampling frequency
        order (int): Order of the bandpass filter

    Returns:
        numpy.ndarray: Bandpassed signal
    """

    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype="band", output="sos")
    bandpass_sig = signal.sosfiltfilt(sos, sig)

    return bandpass_sig


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

    overlap = sn - ceil((ML - sn) / (current_len - sn))

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

    overlap = fl - floor(current_len / fn)

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
