import os
import tempfile

import ffmpeg
import numpy as np
from scipy import signal
from scipy.io import wavfile


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

    # h = signal.firwin(N, Wn, pass_zero=False, fs=sampling_rate)  # FIR Filter Design
    h = signal.firwin(
        order, Wn, width=0.0012, window="hann", pass_zero=False, scale=True, fs=sampling_rate
    )  # FIR Filter Design
    sig_len = len(sig)  # Length of the signal

    # Filter the signal with zero-padding for shorter signals
    if sig_len <= 3 * order:
        pad_length = (
            int(1.5 * order - sig_len // 2) + 50
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
