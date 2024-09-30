import math
import numpy as np
from scipy.fft import fft
from scipy.signal import get_window


# DFT1 instantaneous phase estimation
def segmented_phase_estimation_DFT1(
    s_in, sampling_rate, nominal_enf, N_DFT, step_size, window_len
):
    """
    Estimates the instantaneous phase of an input signal using DFT1 for segments.

    Args:
        s_in (numpy.ndarray): The input signal for phase estimation
        sampling_rate (float): The sampling rate of the input signal
        nominal_enf (float): The nominal electrical network frequency
        N_DFT (int): The number of points for the DFT
        step_size (int): The number of samples to shift between segments
        window_len (int): The length of each segment to analyze

    Returns:
        numpy.ndarray: An array of estimated phases corresponding to each segment of the input signal.
    """
    segments = []

    step_size = 20
    window_len = 200

    for i in range(0, len(s_in), int(step_size)):
        segments.append(s_in[i : i + int(window_len)])

    phases = []
    for segment in segments:
        phase = phase_estimation_DFT1(segment, sampling_rate, N_DFT, nominal_enf)
        phases.append(phase)

    phases = [2 * (x + np.pi / 2) for x in phases]
    phases = np.unwrap(phases)
    phases = [(x / 2.0 - np.pi / 2) for x in phases]

    return phases


# Estimate frequency and phase with DFT
def phase_estimation_DFT0(s_tone, sampling_rate, N_DFT):
    """
    Estimate frequency and phase of a signal using the Discrete Fourier Transform (DFT).

    Args:
        s_tone (np.ndarray): The input signal (time-domain)
        sampling_rate (float): Sampling rate
        N_DFT (int): The number of points in the DFT (zero-padding length & frequency resolution)

    Returns:
        tuple: Estimated frequency (float) and phase (float) of the input signal.
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
    f0_estimated = k_max * (sampling_rate) / (N_DFT)  # estimated frequency of the single tone

    # Estimate the phase
    phi0_estimated = np.angle(X[k_max])  # Argument (angle) of the DFT function

    return f0_estimated, phi0_estimated


def phase_estimation_DFT1(s_tone, sampling_rate, N_DFT, f0_estimated):
    """
    Estimates the instantaneous phase of a tone using the Discrete Fourier Transform (DFT).

    Args:
        s_tone (numpy.ndarray): The input signal (tone) for phase estimation
        sampling_rate (float): The sampling rate of the input signal
        N_DFT (int): The number of points for the DFT
        f0_estimated (float): The estimated frequency of the tone

    Returns:
        float: The estimated instantaneous phase of the tone

    Process:
        - Applies a Hann window to the input signal's derivative.
        - Zero-pads the derivative signal to the specified length (N_DFT).
        - Computes the DFT of the windowed derivative signal.
        - Calculates the corresponding DFT index for the estimated frequency.
        - Estimates the phase based on the DFT of the derivative and the estimated frequency.
        - Compares the estimated phase with a phase calculated using a previous method (DFT⁰).
        - Returns the phase that is closest to the DFT⁰ estimate.
    """

    # ......Estimate the frequency......#
    window_type = "hann"
    M = len(s_tone)

    # Get the window type
    window = get_window(window_type, M - 1)

    # Calculate the approx. first derivative of single tone
    s_tone_diff = sampling_rate * np.diff(s_tone)

    # Windowing
    s_tone_diff_windowed = s_tone_diff * window

    # Zero-Padding of the signal
    s_tone_padded_diff = np.pad(s_tone_diff_windowed, (0, N_DFT - M), "constant")

    # Calculate the DFT
    X_diff = fft(s_tone_padded_diff, n=N_DFT)

    k_DFT = (N_DFT * f0_estimated) / sampling_rate

    # Validate the frequency result
    _, phi_DFT0 = phase_estimation_DFT0(
        s_tone, sampling_rate, N_DFT
    )  # Calculate phase with DFT⁰ method to compare the values

    omega_0 = 2 * np.pi * f0_estimated / sampling_rate
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
