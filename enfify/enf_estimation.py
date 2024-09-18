"""Module for ENF frequency and phase estimation"""

import librosa
import math
import numpy as np
from scipy.fft import fft
from scipy.signal import get_window, hilbert
from numba import jit


# Estimate frequency and phase with DFT⁰ ((Rodriguez Paper)
def phase_estimation_DFT0(s_tone, Fs, N_DFT):
    """_summary_

    Args:
        s_tone (_type_): _description_
        Fs (_type_): _description_
        N_DFT (_type_): _description_

    Returns:
        _type_: _description_
    """

    window_type = "hann"
    M = len(s_tone)
    window = get_window(window_type, M)
    s_tone = s_tone * window

    # Zero-pad the signal to length N_DFT
    s_tone_padded = np.pad(s_tone, (0, N_DFT - M), "constant")

    # Compute the DFT
    X = fft(s_tone_padded, N_DFT)

    # Find the peak in the magnitude spectrum
    magnitude_spectrum = np.abs(X)  # Magnitude of the DFT (Amplitude)
    k_max = np.argmax(magnitude_spectrum)  # Maximum Amplitude
    f0_estimated = k_max * (Fs) / (N_DFT)  # estimated frequency of the single tone

    # Estimate the phase
    phi0_estimated = np.angle(X[k_max])  # Argument (angle) of the DFT function

    return f0_estimated, phi0_estimated


# Estimate frequency with DFT¹ instantaneous estimation (Rodriguez Paper)
def freq_estimation_DFT1(s_tone, Fs, N_DFT):
    """_summary_

    Args:
        s_tone (_type_): _description_
        Fs (_type_): _description_
        N_DFT (_type_): _description_

    Returns:
        _type_: _description_
    """

    # ......Estimate the frequency......#
    window_type = "hann"
    M = len(s_tone)

    # Get the window type
    window = get_window(window_type, M - 1)

    # Calculate the approx. first derivative of single tone
    s_tone_diff = Fs * np.diff(s_tone)
    s_tone = s_tone[1:]

    # Windowing
    s_tone_windowed = s_tone * window
    s_tone_diff_windowed = s_tone_diff * window

    # Zero-Padding of the signal
    s_tone_padded = np.pad(s_tone_windowed, (0, N_DFT - M), "constant")
    s_tone_padded_diff = np.pad(s_tone_diff_windowed, (0, N_DFT - M), "constant")

    # Calculate the DFT
    X = fft(s_tone_padded, n=N_DFT)
    X_diff = fft(s_tone_padded_diff, n=N_DFT)

    # Compute the amplitude spectrum and max. amplitude
    abs_X = np.abs(X)
    k_max = np.argmax(abs_X)
    abs_X_diff = np.abs(X_diff)

    # Estimated frequency of the single tone
    F_kmax = (np.pi * k_max) / (N_DFT * np.sin(np.pi * k_max / N_DFT))
    f0_estimated = (F_kmax * abs_X_diff[k_max]) / (2 * np.pi * abs_X[k_max])

    # Validate the frequency result
    k_DFT = (N_DFT * f0_estimated) / Fs
    try:
        k_DFT >= (k_max - 0.5) and k_DFT < (k_max + 0.5)
    except ValueError:
        print("estimated frequency is not valid")

    return f0_estimated


# Estimate phase with DFT¹ instantaneous estimation (Rodriguez Paper)
def phase_estimation_DFT1(s_tone, Fs, N_DFT, f0_estimated):
    """_summary_

    Args:
        s_tone (_type_): _description_
        Fs (_type_): _description_
        N_DFT (_type_): _description_
        f0_estimated (_type_): _description_

    Returns:
        _type_: _description_
    """

    # ......Estimate the frequency......#
    window_type = "hann"
    M = len(s_tone)

    # Get the window type
    window = get_window(window_type, M - 1)

    # Calculate the approx. first derivative of single tone
    s_tone_diff = Fs * np.diff(s_tone)

    # Windowing
    s_tone_diff_windowed = s_tone_diff * window

    # Zero-Padding of the signal
    s_tone_padded_diff = np.pad(s_tone_diff_windowed, (0, N_DFT - M), "constant")

    # Calculate the DFT
    X_diff = fft(s_tone_padded_diff, n=N_DFT)

    k_DFT = (N_DFT * f0_estimated) / Fs

    # Validate the frequency result
    _, phi_DFT0 = phase_estimation_DFT0(
        s_tone, Fs, N_DFT
    )  # Calculate phase with DFT⁰ method to compare the values

    omega_0 = 2 * np.pi * f0_estimated / Fs
    k_low = math.floor(k_DFT)
    k_high = math.ceil(k_DFT)

    # Handle the case where k_low == k_high
    if k_low == k_high:
        theta = np.angle(X_diff[k_low])
    else:
        theta_low = np.angle(X_diff[k_low])
        theta_high = np.angle(X_diff[k_high])
        theta = (k_DFT - k_low) * (theta_high - theta_low) / (k_high - k_low) + theta_low

    numerator = np.tan(theta) * (1 - np.cos(omega_0)) + np.sin(omega_0)
    denominator = 1 - np.cos(omega_0) - np.tan(theta) * np.sin(omega_0)
    phase_estimated = np.arctan(numerator / denominator)

    # Calculate both possible values of phi and compare them
    phi_1 = phase_estimated
    phi_2 = phase_estimated + np.pi if np.arctan(phase_estimated) >= 0 else phase_estimated - np.pi

    if abs(phi_1 - phi_DFT0) < abs(phi_2 - phi_DFT0):  # compare with phi calculated via DFT⁰
        phi = phi_1
    else:
        phi = phi_2

    return phi


# ..... Estimate the instantaneous frequency of a tone in


def segmented_freq_estimation_DFT1(s_in, f_s, N_DFT, step_size, window_len):
    """_summary_

    Args:
        s_in (_type_): _description_
        f_s (_type_): _description_
        num_cycles (_type_): _description_
        N_DFT (_type_): _description_
        nominal_enf (_type_): _description_

    Returns:
        _type_: _description_
    """
    segments = []

    for i in range(0, len(s_in), step_size):
        segments.append(s_in[i : i + window_len])

    freqs = []
    for i in range(len(segments)):
        freq = freq_estimation_DFT1(segments[i], f_s, N_DFT)
        freqs.append(freq)

    freqs = np.array(freqs)

    return freqs


def segmented_phase_estimation_DFT1(s_in, f_s, nominal_enf, N_DFT, step_size, window_len):
    """_summary_

    Args:
        s_in (_type_): _description_
        f_s (_type_): _description_
        num_cycles (_type_): _description_
        N_DFT (_type_): _description_
        nominal_enf (_type_): _description_

    Returns:
        _type_: _description_
    """
    segments = []

    for i in range(0, len(s_in), step_size):
        segments.append(s_in[i : i + window_len])

    phases = []
    for segment in segments:
        phase = phase_estimation_DFT1(segment, f_s, N_DFT, nominal_enf)
        phases.append(phase)

    phases = [2 * (x + np.pi / 2) for x in phases]
    phases = np.unwrap(phases)
    phases = [(x / 2.0 - np.pi / 2) for x in phases]

    return phases


# Hilbert instantaneous phase estimation
def hilbert_instantaneous_phase(signal):
    """_summary_

    Args:
        signal (_type_): _description_

    Returns:
        _type_: _description_
    """
    analytic_sig = hilbert(signal)
    inst_phase = np.unwrap(np.angle(analytic_sig))
    return inst_phase


# Hilbert segmented phase estimation
def segmented_phase_estimation_hilbert_new(s_in, step_size, window_len):
    """_summary_

    Args:
        s_in (_type_): _description_
        f_s (_type_): _description_
        num_cycles (_type_): _description_
        nominal_enf (_type_): _description_

    Returns:
        _type_: _description_
    """

    window_type = "hann"

    segments = []

    for i in range(0, len(s_in), step_size):
        segments.append(s_in[i : i + window_len])

    phases = []

    for i in range(len(segments)):

        M = len(segments[i])
        window = get_window(window_type, M)
        hann_segment = segments[i] * window

        phase = hilbert_instantaneous_phase(hann_segment)
        phase = np.mean(phase)
        phases.append(phase)

    phases = np.unwrap(phases)
    phases = np.array(phases)

    return phases


# Hilbert segmented phase estimation
def segmented_phase_estimation_hilbert(s_in, f_s, num_cycles, nominal_enf):
    """_summary_

    Args:
        s_in (_type_): _description_
        f_s (_type_): _description_
        num_cycles (_type_): _description_
        nominal_enf (_type_): _description_

    Returns:
        _type_: _description_
    """

    window_type = "hann"

    step_size = int(f_s // nominal_enf)

    num_blocks = len(s_in) // step_size - (num_cycles - 1)

    segments = [s_in[i * step_size : (i + num_cycles) * step_size] for i in range(num_blocks)]

    phases = []

    for i in range(len(segments)):

        M = len(segments[i])
        window = get_window(window_type, M)
        hann_segment = segments[i] * window

        phase = hilbert_instantaneous_phase(hann_segment)
        phase = np.mean(phase)
        phases.append(phase)

    phases = np.unwrap(phases)
    phases = np.array(phases)

    return phases


# STFT
@jit(nopython=True)
def principal_argument(v):
    """Principal argument function

    Args:
        v (float or np.ndarray): Value (or vector of values)

    Returns:
        w (float or np.ndarray): Principle value of v
    """
    w = np.mod(v + 0.5, 1) - 0.5
    return w


@jit(nopython=True)
def compute_if(X, Fs, N, H):
    """Instantenous frequency (IF) estamation

    Args:
        X (np.ndarray): STFT
        Fs (scalar): Sampling rate
        N (int): Window size in samples
        H (int): Hop size in samples

    Returns:
        F_coef_IF (np.ndarray): Matrix of IF values
    """
    phi_1 = np.angle(X[:, 0:-1]) / (2 * np.pi)
    phi_2 = np.angle(X[:, 1:]) / (2 * np.pi)

    K = X.shape[0]
    index_k = np.arange(0, K).reshape(-1, 1)
    # Bin offset (FMP, Eq. (8.45))
    kappa = (N / H) * principal_argument(phi_2 - phi_1 - index_k * H / N)
    # Instantaneous frequencies (FMP, Eq. (8.44))
    F_coef_IF = (index_k + kappa) * Fs / N

    # Extend F_coef_IF by copying first column to match dimensions of X
    F_coef_IF = np.hstack((np.copy(F_coef_IF[:, 0]).reshape(-1, 1), F_coef_IF))

    return F_coef_IF


def compute_max_energy_frequency_over_time(X, F_coef_IF):
    """Compute the dominant frequency over time based on max energy.

    Args:
        X (np.ndarray): STFT matrix (complex-valued).
        F_coef_IF (np.ndarray): Instantaneous frequency matrix.

    Returns:
        max_energy_freq (np.ndarray): 1D array of dominant frequencies over time.
    """
    # Magnitude of STFT values (|X|)
    magnitude = np.abs(X)

    # Find the index of the frequency bin with maximum magnitude in each time frame
    max_energy_indices = np.argmax(magnitude, axis=0)

    # Use these indices to extract the corresponding instantaneous frequencies
    max_energy_freq = F_coef_IF[max_energy_indices, np.arange(F_coef_IF.shape[1])]

    return max_energy_freq


def STFT(signal, sampling_freq, step_size, window_len):
    """
    Computes the Short-Time Fourier Transform (STFT) of a given signal, extracts instantaneous frequency, and computes
    the dominant frequency over time.

    Args:
        signal (numpy.ndarray): The input time-domain signal (1D array).
        sampling_freq (int or float): The sampling frequency of the signal (in Hz).
        step_size (int): Hop length, the number of samples between successive frames.
        window_len (int): Length of each FFT window (in samples).

    Returns:
        freqs (numpy.ndarray): Array of dominant frequencies over time (in Hz).
    """
    # Compute the Short-Time Fourier Transform (STFT)
    X = librosa.stft(
        signal, n_fft=window_len, hop_length=step_size, win_length=window_len, window="hamming"
    )

    # Compute the instantaneous frequency (placeholder function)
    F_coef_IF = compute_if(X, sampling_freq, window_len, step_size)

    # Compute the dominant frequency over time (placeholder function)
    freqs = compute_max_energy_frequency_over_time(X, F_coef_IF)

    return freqs
