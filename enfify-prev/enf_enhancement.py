"""Module for ENF enhancement"""

# TODO: Add paper DOI links in docstrings of functions.

import numpy as np
import math

from enf_estimation import segmented_freq_estimation_DFT1
from tqdm import tqdm
from numba import jit, prange
from scipy.signal import get_window, windows
from scipy.fft import fft

# import matplotlib.pyplot as plt
from enf_estimation import STFT


# ...........................RFA................................#


@jit(nopython=True, fastmath=True)
def z_SFM(sig, n, sampling_rate, alpha, tau):
    """Computes the z_SFM value with JIT optimization.

    Args:
        sig (numpy.ndarray): Audio signal
        n (int): The index in the signal array at which to compute the z_SFM value
        sampling_rate (float): The sampling frequency of the input signal
        alpha (float): The scaling factor used in the phase calculation
        tau (int): The number of samples to include before and after the index n for summation

    Returns:
        complex: The computed z_SFM value as a complex number, representing the phase shift
    """
    sum_sig = np.sum(sig[n - tau : n + tau + 1])
    z = np.exp(1j * 2 * np.pi * (1 / sampling_rate) * alpha * sum_sig)
    return z


@jit(nopython=True, parallel=True, fastmath=True)
def kernel_function(sig, f, n, sampling_rate, alpha, tau_values, tau_dash_values):
    """Computes the kernel function using JIT and vectorized operations.

    Args:
        sig (numpy.ndarray): Audio signal
        f (float): The frequency at which to evaluate the kernel
        n (int): The index in the signal at which the kernel is computed
        sampling_rate (float): The sampling frequency of the input signal
        alpha (float): The scaling factor used in the kernel calculation
        tau_values (numpy.ndarray): Array of tau values for the first part of the kernel
        tau_dash_values (numpy.ndarray): Array of tau values for the second part of the kernel

    Returns:
        numpy.ndarray: The angles of the computed kernel function values as a complex number array
    """
    auto_corr = np.empty(len(tau_values), dtype=np.complex128)
    auto_corr_dash = np.empty(len(tau_dash_values), dtype=np.complex128)

    for i in range(len(tau_values)):
        auto_corr[i] = z_SFM(sig, n, sampling_rate, alpha, tau_values[i])
        auto_corr_dash[i] = z_SFM(sig, n, sampling_rate, alpha, tau_dash_values[i])

    sin_vals = np.sin(2 * np.pi * (1 / sampling_rate) * f * tau_values)
    cos_vals = np.cos(2 * np.pi * (1 / sampling_rate) * f * tau_values)

    # Precompute exponents to save time in kernel calculation
    kernel = (auto_corr**sin_vals) * (auto_corr_dash**cos_vals)
    return np.angle(kernel)


@jit(nopython=True, parallel=True, fastmath=True)
def rfa_kernel_phases(
    sig, denoised_sig, Nx, f_start, sampling_rate, alpha, tau, tau_values, tau_dash_values
):
    """
    Computes the denoised signal using RFA kernel phases.

    Args:
        sig (numpy.ndarray): Audio Signal
        denoised_sig (numpy.ndarray): Denoised audio signal
        Nx (int): The number of frequency bins
        f_start (numpy.ndarray): The starting frequencies for each bin
        sampling_rate (float): The sampling frequency of the input signal
        alpha (float): The scaling factor for the kernel
        tau (int): The parameter influencing the kernel width
        tau_values (numpy.ndarray): Array of tau values for the kernel computation
        tau_dash_values (numpy.ndarray): Array of tau dash values for the kernel computation

    Returns:
        numpy.ndarray: The computed denoised signal
    """
    for n in prange(Nx - 1):
        f = f_start[n]
        kernel_phases = kernel_function(
            sig, f, n, sampling_rate, alpha, tau_values, tau_dash_values
        )
        denoised_sig[n] = np.sum(kernel_phases) / ((tau + 1) * tau * np.pi * alpha)

    return denoised_sig


def RFA(sig, sampling_rate, tau, epsilon, var_I, estimated_enf):
    """Recursive Frequency Adaptation algorithm with partial JIT optimization.

    Args:
        sig (numpy.ndarray): Audio signal
        sampling_rate (float): The sampling frequency of the input signal
        tau (int): The parameter influencing the kernel widt
        epsilon (float): Convergence threshold for frequency adaptation
        var_I (int): Number of iterations for the adaptation process
        estimated_enf (float): Estimated ENF frequency

    Returns:
        numpy.ndarray: The denoised signal after applying RFA
    """
    Nx = len(sig)
    alpha = 1 / (4 * sampling_rate * np.max(sig))
    f_start = estimated_enf * np.ones(Nx)
    tau_values = np.arange(1, tau + 1)
    tau_dash_values = tau_values + int(np.round(sampling_rate / (4 * estimated_enf)))

    for k in tqdm(range(var_I)):
        denoised_sig = np.zeros(Nx)

        denoised_sig = rfa_kernel_phases(
            sig, denoised_sig, Nx, f_start, sampling_rate, alpha, tau, tau_values, tau_dash_values
        )

        # Peak frequency estimation
        peak_freqs = segmented_freq_estimation_DFT1(
            denoised_sig, sampling_rate, num_cycles=100, N_DFT=20_000, nominal_enf=estimated_enf
        )

        base_repeats = Nx // len(peak_freqs)
        remainder = Nx % len(peak_freqs)
        repeat_counts = np.full(len(peak_freqs), base_repeats)
        repeat_counts[:remainder] += 1
        new_freqs = np.repeat(peak_freqs, repeat_counts)

        f_diff = new_freqs - f_start
        f_start = new_freqs

        val = np.sum(f_diff**2) / np.sum(f_start**2)

        if val <= epsilon:
            return denoised_sig

        sig = denoised_sig  # Update the signal for the next iteration

    return denoised_sig


## Use DFT1 as STFT method for the instantaneous frequency estimation
def RFA_DFT1(sig, sampling_rate, tau, var_I, estimated_enf, window_len, step_size, N_DFT):
    """Recursive Frequency Adaptation algorithm with partial JIT optimization and DFT1 instantaneous frequency estimation.

    Args:
        sig (numpy.ndarray): Audio signal
        sampling_rate (float): The sampling frequency of the input signal
        tau (int): The parameter influencing the kernel widt
        epsilon (float): Convergence threshold for frequency adaptation
        var_I (int): Number of iterations for the adaptation process
        estimated_enf (float): Estimated ENF frequency

    Returns:
        numpy.ndarray: The denoised signal after applying RFA
    """
    Nx = len(sig)
    alpha = 1 / (4 * sampling_rate * np.max(sig))
    f_start = estimated_enf * np.ones(Nx)
    tau_values = np.arange(1, tau + 1)
    tau_dash_values = tau_values + int(np.round(sampling_rate / (4 * estimated_enf)))

    for k in tqdm(range(var_I)):
        denoised_sig = np.zeros(Nx)

        denoised_sig = rfa_kernel_phases(
            sig, denoised_sig, Nx, f_start, sampling_rate, alpha, tau, tau_values, tau_dash_values
        )

        # Peak frequency estimation
        peak_freqs = segmented_freq_estimation_DFT1(
            denoised_sig, sampling_rate, N_DFT, step_size, window_len
        )

        base_repeats = Nx // len(peak_freqs)
        remainder = Nx % len(peak_freqs)
        repeat_counts = np.full(len(peak_freqs), base_repeats)
        repeat_counts[:remainder] += 1
        new_freqs = np.repeat(peak_freqs, repeat_counts)

        f_start = new_freqs

        sig = denoised_sig  # Update the signal for the next iteration

    return sig


## Use STFT
def RFA_STFT(sig, sampling_rate, tau, var_I, estimated_enf, window_len, step_size):
    """
    Recursive Frequency Adaptation (RFA) algorithm using STFT for frequency estimation

    Args:
        sig (numpy.ndarray): The input signal array
        sampling_rate (float): The sampling frequency of the input signal
        tau (int): The parameter influencing the kernel width
        var_I (int): Number of iterations for the adaptation process
        estimated_enf (float): Estimated ENF frequency
        window_len (int): Length of the window for STFT
        step_size (int): Step size for windowing in STFT

    Returns:
        numpy.ndarray: The denoised signal after applying RFA with STFT
    """
    Nx = len(sig)
    alpha = 1 / (4 * sampling_rate * np.max(sig))
    f_start = estimated_enf * np.ones(Nx)
    tau_values = np.arange(1, tau + 1)
    tau_dash_values = tau_values + int(np.round(sampling_rate / (4 * estimated_enf)))

    for k in tqdm(range(var_I)):
        denoised_sig = np.zeros(Nx)

        denoised_sig = rfa_kernel_phases(
            sig, denoised_sig, Nx, f_start, sampling_rate, alpha, tau, tau_values, tau_dash_values
        )

        # Peak frequency estimation via STFT
        peak_freqs = STFT(denoised_sig, sampling_rate, step_size, window_len)

        base_repeats = Nx // len(peak_freqs)
        remainder = Nx % len(peak_freqs)
        repeat_counts = np.full(len(peak_freqs), base_repeats)
        repeat_counts[:remainder] += 1
        new_freqs = np.repeat(peak_freqs, repeat_counts)

        f_start = new_freqs

        sig = denoised_sig  # Update the signal for the next iteration

    return sig


# ...................................Variational Mode Decomposition...................................#


def VMD(signal, alpha, tau, num_modes, enforce_DC, tolerance):
    """Variational Mode Decomposition (VMD)

    Args:
        signal (array): Input signal
        alpha (float): Regularization parameter for the modes
        tau (float): Time-step for dual ascent
        num_modes (int): Number of modes to extract
        enforce_DC (bool): Whether to enforce a DC component
        tolerance (float): Convergence tolerance

    Returns:
        modes (array): Decomposed modes
        mode_spectra_final (array): Final mode spectra
        final_freq_centers (array): Center frequencies of modes
    """

    # Mirror signal at boundaries
    if len(signal) % 2:
        midpoint = math.ceil(len(signal) / 2)
        left_mirror = np.concatenate((np.flipud(signal[: midpoint - 1]), signal), axis=0)
        mirrored_signal = np.concatenate((left_mirror, np.flipud(signal[midpoint:])), axis=0)
    else:
        midpoint = len(signal) // 2
        left_mirror = np.concatenate((np.flipud(signal[:midpoint]), signal), axis=0)
        mirrored_signal = np.concatenate((left_mirror, np.flipud(signal[midpoint:])), axis=0)

    # Define time and frequency domains
    total_length = len(mirrored_signal)
    time_domain = np.arange(1, total_length + 1) / total_length
    spectral_domain = time_domain - 0.5 - (1 / total_length)

    # Iteration parameters
    max_iterations = 500
    mode_alphas = alpha * np.ones(num_modes)

    # FFT of the mirrored signal and preparation for iterations
    signal_spectrum = np.fft.fftshift(np.fft.fft(mirrored_signal))
    positive_spectrum = np.copy(signal_spectrum)
    positive_spectrum[: total_length // 2] = 0

    # Initialize frequency center estimates
    freq_centers = np.zeros((max_iterations, num_modes))
    for mode in range(num_modes):
        freq_centers[0, mode] = (0.5 / num_modes) * mode

    # Enforce DC mode if required
    if enforce_DC:
        freq_centers[0, 0] = 0

    # Initialize dual variables and other parameters
    dual_vars = np.zeros((max_iterations, len(spectral_domain)), dtype=complex)
    convergence_criteria = tolerance + np.spacing(1)
    iteration_count = 0
    mode_sum = 0
    mode_spectra = np.zeros((max_iterations, len(spectral_domain), num_modes), dtype=complex)

    # Main iterative update loop
    while convergence_criteria > tolerance and iteration_count < max_iterations - 1:
        mode_sum = (
            mode_spectra[iteration_count, :, num_modes - 1]
            + mode_sum
            - mode_spectra[iteration_count, :, 0]
        )

        mode_spectra[iteration_count + 1, :, 0] = (
            positive_spectrum - mode_sum - dual_vars[iteration_count, :] / 2
        ) / (1 + mode_alphas[0] * (spectral_domain - freq_centers[iteration_count, 0]) ** 2)

        if not enforce_DC:
            freq_centers[iteration_count + 1, 0] = np.dot(
                spectral_domain[total_length // 2 :],
                abs(mode_spectra[iteration_count + 1, total_length // 2 :, 0]) ** 2,
            ) / np.sum(abs(mode_spectra[iteration_count + 1, total_length // 2 :, 0]) ** 2)

        for mode in range(1, num_modes):
            mode_sum = (
                mode_spectra[iteration_count + 1, :, mode - 1]
                + mode_sum
                - mode_spectra[iteration_count, :, mode]
            )
            mode_spectra[iteration_count + 1, :, mode] = (
                positive_spectrum - mode_sum - dual_vars[iteration_count, :] / 2
            ) / (
                1
                + mode_alphas[mode] * (spectral_domain - freq_centers[iteration_count, mode]) ** 2
            )
            freq_centers[iteration_count + 1, mode] = np.dot(
                spectral_domain[total_length // 2 :],
                abs(mode_spectra[iteration_count + 1, total_length // 2 :, mode]) ** 2,
            ) / np.sum(abs(mode_spectra[iteration_count + 1, total_length // 2 :, mode]) ** 2)

        dual_vars[iteration_count + 1, :] = dual_vars[iteration_count, :] + tau * (
            np.sum(mode_spectra[iteration_count + 1, :, :], axis=1) - positive_spectrum
        )

        iteration_count += 1

        convergence_criteria = np.spacing(1)
        for mode in range(num_modes):
            convergence_criteria += (1 / total_length) * np.dot(
                (
                    mode_spectra[iteration_count, :, mode]
                    - mode_spectra[iteration_count - 1, :, mode]
                ),
                np.conj(
                    mode_spectra[iteration_count, :, mode]
                    - mode_spectra[iteration_count - 1, :, mode]
                ),
            )

        convergence_criteria = np.abs(convergence_criteria)

    # Postprocessing to extract modes and their spectra
    max_iterations = min(max_iterations, iteration_count)
    final_freq_centers = freq_centers[:max_iterations, :]

    half_idxs = np.flip(np.arange(1, total_length // 2 + 1), axis=0)
    final_mode_spectra = np.zeros((total_length, num_modes), dtype=complex)
    final_mode_spectra[total_length // 2 : total_length, :] = mode_spectra[
        max_iterations - 1, total_length // 2 : total_length, :
    ]
    final_mode_spectra[half_idxs, :] = np.conj(
        mode_spectra[max_iterations - 1, total_length // 2 : total_length, :]
    )
    final_mode_spectra[0, :] = np.conj(final_mode_spectra[-1, :])

    modes = np.zeros((num_modes, len(time_domain)))
    for mode in range(num_modes):
        modes[mode, :] = np.real(np.fft.ifft(np.fft.ifftshift(final_mode_spectra[:, mode])))

    modes = modes[:, total_length // 4 : 3 * total_length // 4]

    mode_spectra_final = np.zeros((modes.shape[1], num_modes), dtype=complex)
    for mode in range(num_modes):
        mode_spectra_final[:, mode] = np.fft.fftshift(np.fft.fft(modes[mode, :]))

    return modes, mode_spectra_final, final_freq_centers


# ...................................Maximum Likelyhood estimators, maybe for the future...................................#


def stft_search(sig, sampling_rate, win_dur, step_dur, fc, bnd, fft_fac):
    """
    Perform STFT-based instantaneous frequency search.

    Args:
        sig (numpy.ndarray): Input signal
        sampling_rate (float): Sampling frequency of the signal
        win_dur (float): Duration of the window for STFT
        step_dur (float): Step duration for sliding the window
        fc (numpy.ndarray): Center frequencies of interest
        bnd (numpy.ndarray): Bandwidths for frequency estimation
        fft_fac (float): FFT resolution factor

    Returns:
        numpy.ndarray: Instantaneous frequencies detected in the signal.
    """
    win_len = int(win_dur * sampling_rate)
    win_func = windows.boxcar(win_len)
    step = int(step_dur * sampling_rate)
    NFFT = int(fft_fac * sampling_rate)

    win_pos = np.arange(0, len(sig) - win_len + 1, step)
    IF = np.zeros(len(win_pos))

    # Set search region
    search_1st = np.arange(round((50 - bnd[0] / 2) * fft_fac), round((50 + bnd[0] / 2) * fft_fac))
    len_band = len(search_1st)
    search_reg = np.kron(search_1st, (fc / 50)).astype(int)

    # Search loop
    for i, pos in enumerate(win_pos):
        temp_fft = fft(sig[pos : pos + win_len] * win_func, n=NFFT)
        half_fft = temp_fft[: NFFT // 2]
        abs_half_fft = np.abs(half_fft)

        fbin_cand = abs_half_fft[search_reg]
        fbin_cand = fbin_cand.reshape((len(fc), len_band))
        weighted_fbin = np.sum(fbin_cand**2, axis=0)
        max_val = np.max(weighted_fbin)
        peak_loc = search_1st[np.where(weighted_fbin == max_val)[0][0]]
        IF[i] = peak_loc * sampling_rate / NFFT * 2

    norm_fc = 100
    norm_bnd = 100 * bnd[0] / fc[0]
    IF[IF < norm_fc - norm_bnd] = norm_fc - norm_bnd
    IF[IF > norm_fc + norm_bnd] = norm_fc + norm_bnd

    return IF


def func_STFT_multi_tone_search_weighted(
    signal, sampling_rate, window_dur, step_size_dur, fc, bound, FFT_res_factor
):
    """
    Perform multi-tone frequency search using STFT with harmonic weighting.

    Args:
        signal (numpy.ndarray): Input signal
        sampling_rate (float): Sampling frequency of the signal
        window_dur (float): Duration of the window for STFT
        step_size_dur (float): Step duration for sliding the window
        fc (numpy.ndarray): Center frequencies of interest
        bound (numpy.ndarray): Bandwidth for frequency estimation
        FFT_res_factor (float): FFT resolution factor

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Instantaneous frequencies and weights for each frequency component.
    """
    window_length = int(window_dur * sampling_rate)
    window_func = get_window("boxcar", window_length)
    step_size = int(step_size_dur * sampling_rate)
    NFFT = int(FFT_res_factor * sampling_rate)
    window_pos = np.arange(0, len(signal) - window_length + 1, step_size)
    IF = np.zeros(len(window_pos))  # output IF without interpolation

    # Set bandwidth to estimate local SNR according to [2]
    band_low_signal_1st = round(49.98 * FFT_res_factor)
    band_high_signal_1st = round(50.02 * FFT_res_factor)  # signal band: 49.98~50.02 Hz
    band_low_noise_1st = round(49.9 * FFT_res_factor)
    band_high_noise_1st = round(
        50.1 * FFT_res_factor
    )  # noise band: 49.9~50.1 Hz excluding signal band

    weights = np.zeros(len(fc))

    # Set harmonic search region
    search_region_1st = np.arange(
        round((50 - bound[0] / 2) * FFT_res_factor), round((50 + bound[0] / 2) * FFT_res_factor)
    )
    length_per_band = len(search_region_1st)
    search_region = np.kron(search_region_1st, (fc / 50)).astype(int)

    # Search loop
    for i, pos in enumerate(window_pos):
        temp = np.fft.fft(signal[pos : pos + window_length] * window_func, NFFT)
        HalfTempFFT = temp[: NFFT // 2]
        absHalfTempFFT = np.abs(HalfTempFFT)

        # Calculate weights for each harmonic component according to [2]
        for j in range(len(fc)):
            signal_component = absHalfTempFFT[
                band_low_signal_1st * (fc[j] // 50) : band_high_signal_1st * (fc[j] // 50)
            ]
            signal_plus_noise = absHalfTempFFT[
                band_low_noise_1st * (fc[j] // 50) : band_high_noise_1st * (fc[j] // 50)
            ]
            weights[j] = np.linalg.norm(signal_component) ** 2 / (
                np.linalg.norm(signal_plus_noise) ** 2 - np.linalg.norm(signal_component) ** 2
            )

        weights = weights / np.linalg.norm(weights)
        fbin_candidate = absHalfTempFFT[search_region]
        fbin_candidate = np.diag(weights) @ fbin_candidate.reshape(len(fc), length_per_band)
        weighted_fbin = np.sum(
            fbin_candidate**2, axis=0
        )  # harmonically weighted frequency bin energy
        # ValueMax = np.max(weighted_fbin)
        PeakLoc = search_region_1st[np.argmax(weighted_fbin)]
        IF[i] = (
            PeakLoc * sampling_rate / NFFT * 2
        )  # (one sample shift compensated, location no need to "- 1")

    norm_fc = 100
    norm_bound = 100 * bound[0] / fc[0]
    IF[IF < norm_fc - norm_bound] = norm_fc - norm_bound
    IF[IF > norm_fc + norm_bound] = norm_fc + norm_bound

    return IF, weights
