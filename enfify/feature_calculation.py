import numpy as np
from scipy.fft import fft
from scipy.signal import get_window


def framing(sig, sample_freq, frame_len, frame_shift, window_type):
    """Split signal into frames. Each frame is windowed using the specified window type.
    Exact frame length and shift.

    Args:
        sig (numpy.ndarray): Signal to be split into frames.
        sample_freq (float): Sampling frequency in Hz.
        frame_len (float): Frame length in milliseconds.
        frame_shift (float): Frame shift in milliseconds.
        window_type (str): Window type for windowing the frames.
    """
    # TODO: Error handling for frame_len and frame_shift

    frame_len_samples = int(frame_len / 1000 * sample_freq)
    frame_shift_samples = int(frame_shift / 1000 * sample_freq)

    num_frames = (len(sig) - frame_len_samples + frame_shift_samples) // frame_shift_samples
    frames = np.zeros((num_frames, frame_len_samples))

    window = get_window(window_type, frame_len_samples)
    for i in range(num_frames):
        start = i * frame_shift_samples
        end = start + frame_len_samples
        frames[i] = sig[start:end] * window
    return frames


# Estimate frequency with DFT¹ instantaneous estimation (Rodriguez Paper)
def freq_estimation_DFT1(sig, sample_rate, n_dft, window_type):
    """
    Estimates the instantaneous frequency of a tone using the Discrete Fourier Transform (DFT).

    Args:
        sig (numpy.ndarray): The input signal (tone) for frequency estimation
        sample_rate (float): The sampling rate of the input signal
        n_dft (int): The number of points in the DFT (zero-padding length & frequency resolution)
        window_type (str): The window type for windowing the signal.

    Returns:
        float: The estimated instantaneous frequency of the tone
    """
    # ......Estimate the frequency......#
    M = len(sig)

    # Get the window type
    window = get_window(window_type, M - 1)

    # Calculate the approx. first derivative of single tone
    s_tone_diff = sample_rate * np.diff(sig)
    sig = sig[1:]

    # Windowing
    s_tone_windowed = sig * window
    s_tone_diff_windowed = s_tone_diff * window

    # Zero-Padding of the signal
    s_tone_padded = np.pad(s_tone_windowed, (0, n_dft - M), "constant")
    s_tone_padded_diff = np.pad(s_tone_diff_windowed, (0, n_dft - M), "constant")

    # Calculate the DFT
    X = fft(s_tone_padded, n=n_dft)
    X_diff = fft(s_tone_padded_diff, n=n_dft)

    # Compute the amplitude spectrum and max. amplitude
    abs_X = np.abs(X)
    k_max = np.argmax(abs_X)
    abs_X_diff = np.abs(X_diff)

    # Estimated frequency of the single tone
    F_kmax = (np.pi * k_max) / (n_dft * np.sin(np.pi * k_max / n_dft))
    f0_estimated = (F_kmax * abs_X_diff[k_max]) / (2 * np.pi * abs_X[k_max])

    # Validate the frequency result
    k_DFT = (n_dft * f0_estimated) / sample_rate
    try:
        k_DFT >= (k_max - 0.5) and k_DFT < (k_max + 0.5)
    except ValueError:
        print("estimated frequency is not valid")

    return f0_estimated
