import librosa
import numpy as np
from numba import jit, prange


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
    # Handle odd signal lengths
    if len(signal) % 2:
        signal = signal[:-1]

    # Define time and frequency domains
    total_length = len(signal)
    time_domain = np.arange(1, total_length + 1) / total_length
    spectral_domain = time_domain - 0.5 - (1 / total_length)

    # Iteration parameters
    max_iterations = 500
    mode_alphas = alpha * np.ones(num_modes)

    # FFT of the mirrored signal and preparation for iterations
    signal_spectrum = np.fft.fftshift(np.fft.fft(signal))
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

    for k in range(var_I):
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
def compute_if(X, sampling_rate, N, H):
    """Instantenous frequency (IF) estamation

    Args:
        X (np.ndarray): STFT
        sampling_rate (scalar): Sampling rate
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
    F_coef_IF = (index_k + kappa) * sampling_rate / N

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


def STFT(signal, sampling_rate, step_size, window_len):
    """
    Computes the Short-Time Fourier Transform (STFT) of a given signal, extracts instantaneous frequency, and computes
    the dominant frequency over time.

    Args:
        signal (numpy.ndarray): The input time-domain signal (1D array)
        sampling_rate (int or float): The sampling frequency of the signal (in Hz)
        step_size (int): Hop length, the number of samples between successive frames
        window_len (int): Length of each FFT window (in samples)

    Returns:
        freqs (numpy.ndarray): Array of dominant frequencies over time (in Hz).
    """

    step_size = step_size / 1000 * sampling_rate
    window_len = window_len / 1000 * sampling_rate
    step_size = int(step_size)
    window_len = int(window_len)

    # Compute the Short-Time Fourier Transform (STFT)
    X = librosa.stft(
        signal, n_fft=window_len, hop_length=step_size, win_length=window_len, window="hamming"
    )

    # Compute the instantaneous frequency (placeholder function)
    F_coef_IF = compute_if(X, sampling_rate, window_len, step_size)

    # Compute the dominant frequency over time (placeholder function)
    freqs = compute_max_energy_frequency_over_time(X, F_coef_IF)

    return freqs
