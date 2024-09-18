import numpy as np
from scipy.io import wavfile

from tqdm import tqdm
from numba import jit, prange


import matplotlib.pyplot as plt

# from enfify.preprocessing import cut_signal

# from tqdm import tqdm


from enfify.enf_estimation import (
    segmented_freq_estimation_DFT1,
    segmented_phase_estimation_hilbert,
)
from enfify.preprocessing import (
    downsample_ffmpeg,
    bandpass_filter,
)

from enfify.enf_enhancement import VMD

from scipy.signal import get_window, hilbert

# ...........................RFA................................#


@jit(nopython=True, fastmath=True)
def z_SFM(sig, n, fs, alpha, tau):
    """Computes the z_SFM value with JIT optimization."""
    sum_sig = np.sum(sig[n - tau : n + tau + 1])
    z = np.exp(1j * 2 * np.pi * (1 / fs) * alpha * sum_sig)
    return z


@jit(nopython=True, parallel=True, fastmath=True)
def kernel_function(sig, f, n, fs, alpha, tau_values, tau_dash_values):
    """Computes the kernel function using JIT and vectorized operations."""
    auto_corr = np.empty(len(tau_values), dtype=np.complex128)
    auto_corr_dash = np.empty(len(tau_dash_values), dtype=np.complex128)

    for i in range(len(tau_values)):
        auto_corr[i] = z_SFM(sig, n, fs, alpha, tau_values[i])
        auto_corr_dash[i] = z_SFM(sig, n, fs, alpha, tau_dash_values[i])

    sin_vals = np.sin(2 * np.pi * (1 / fs) * f * tau_values)
    cos_vals = np.cos(2 * np.pi * (1 / fs) * f * tau_values)

    # Precompute exponents to save time in kernel calculation
    kernel = (auto_corr**sin_vals) * (auto_corr_dash**cos_vals)
    return np.angle(kernel)


@jit(nopython=True, parallel=True, fastmath=True)
def rfa_kernel_phases(sig, denoised_sig, Nx, f_start, fs, alpha, tau, tau_values, tau_dash_values):
    for n in prange(Nx - 1):
        f = f_start[n]
        kernel_phases = kernel_function(sig, f, n, fs, alpha, tau_values, tau_dash_values)
        denoised_sig[n] = np.sum(kernel_phases) / ((tau + 1) * tau * np.pi * alpha)

    return denoised_sig


def RFA(sig, fs, tau, epsilon, var_I, estimated_enf):
    """Optimized Recursive Frequency Adaptation algorithm with partial JIT optimization."""
    Nx = len(sig)
    alpha = 1 / (4 * fs * np.max(sig))
    f_start = estimated_enf * np.ones(Nx)
    tau_values = np.arange(1, tau + 1)
    tau_dash_values = tau_values + int(np.round(fs / (4 * estimated_enf)))

    for k in tqdm(range(var_I)):
        denoised_sig = np.zeros(Nx)

        denoised_sig = rfa_kernel_phases(
            sig, denoised_sig, Nx, f_start, fs, alpha, tau, tau_values, tau_dash_values
        )

        # Peak frequency estimation
        peak_freqs = segmented_freq_estimation_DFT1(
            denoised_sig, fs, num_cycles=160, N_DFT=20_000, nominal_enf=estimated_enf
        )

        plt.plot(peak_freqs)
        plt.xlabel("Cycles of nominal enf")
        plt.ylabel("Frequency Hz")
        plt.title("Frequency vs Time")
        plt.grid(True)
        plt.show()

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


def hilbert_instantaneous_freq(signal, fs):
    """_summary_

    Args:
        signal (_type_): _description_
        fs (_type_): _description_

    Returns:
        _type_: _description_
    """
    analytic_sig = hilbert(signal)
    inst_phase = np.unwrap(np.angle(analytic_sig))
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * fs
    inst_freq = np.append(
        inst_freq, inst_freq[-1]
    )  # Diff reduces the number of results by 1 -> dulicate the last frequency
    return inst_freq


def segmented_freq_estimation_hilbert(s_in, f_s, num_cycles, nominal_enf):
    """_summary_

    Args:
        s_in (_type_): _description_
        f_s (_type_): _description_
        num_cycles (_type_): _description_
        nominal_enf (_type_): _description_

    Returns:
        _type_: _description_
    """

    window_type = "hamming"

    step_size = int(f_s // nominal_enf)

    num_blocks = len(s_in) // step_size - (num_cycles - 1)

    segments = [s_in[i * step_size : (i + num_cycles) * step_size] for i in range(num_blocks)]

    freqs = []

    for i in range(len(segments)):

        M = len(segments[i])
        window = get_window(window_type, M)
        hann_segment = segments[i] * window

        freq = hilbert_instantaneous_freq(hann_segment, f_s)
        freq = np.mean(freq)
        freqs.append(freq)

    freqs = np.unwrap(freqs)
    freqs = np.array(freqs)

    return freqs


# ...................................Variational Mode Decomposition 2...................................#
def VMD_2(signal, alpha, tau, num_modes, enforce_DC, tolerance):
    """Variational Mode Decomposition (VMD)

    Args:
        signal (array): Input signal.
        alpha (float): Regularization parameter for the modes.
        tau (float): Time-step for dual ascent.
        num_modes (int): Number of modes to extract.
        enforce_DC (bool): Whether to enforce a DC component.
        tolerance (float): Convergence tolerance.

    Returns:
        modes (array): Decomposed modes.
        mode_spectra_final (array): Final mode spectra.
        final_freq_centers (array): Center frequencies of modes.
    """

    # Mirror signal at boundaries (optimized)
    signal_len = len(signal)
    midpoint = signal_len // 2
    if signal_len % 2:
        mirrored_signal = np.concatenate(
            [np.flipud(signal[:midpoint]), signal, np.flipud(signal[midpoint + 1 :])]
        )
    else:
        mirrored_signal = np.concatenate(
            [np.flipud(signal[:midpoint]), signal, np.flipud(signal[midpoint:])]
        )

    # Time and frequency domains
    total_length = len(mirrored_signal)
    time_domain = np.arange(total_length) / total_length
    spectral_domain = time_domain - 0.5

    # Precompute FFT of the mirrored signal
    signal_spectrum = np.fft.fftshift(np.fft.fft(mirrored_signal))
    positive_spectrum = np.zeros_like(signal_spectrum, dtype=complex)
    positive_spectrum[total_length // 2 :] = signal_spectrum[total_length // 2 :]

    # Mode and frequency center initialization
    max_iterations = 500
    mode_alphas = np.full(num_modes, alpha)
    freq_centers = np.zeros((max_iterations, num_modes))

    # Set initial frequency centers
    freq_centers[0, :] = 0.5 / num_modes * np.arange(num_modes)
    if enforce_DC:
        freq_centers[0, 0] = 0

    # Initialize dual variables and mode spectra
    dual_vars = np.zeros(total_length, dtype=complex)
    mode_spectra = np.zeros((max_iterations, total_length, num_modes), dtype=complex)

    # Iteration parameters
    convergence_criteria = tolerance + np.spacing(1)
    iteration_count = 0

    while convergence_criteria > tolerance and iteration_count < max_iterations - 1:
        mode_sum = np.sum(mode_spectra[iteration_count, :, :], axis=1)

        for mode in range(num_modes):
            residual = positive_spectrum - mode_sum + mode_spectra[iteration_count, :, mode]
            denom = (
                1
                + mode_alphas[mode] * (spectral_domain - freq_centers[iteration_count, mode]) ** 2
            )

            mode_spectra[iteration_count + 1, :, mode] = (residual - dual_vars / 2) / denom

            if mode == 0 and enforce_DC:
                freq_centers[iteration_count + 1, mode] = 0
            else:
                # Update frequency centers using the mode's positive frequencies
                mode_fft_half = mode_spectra[iteration_count + 1, total_length // 2 :, mode]
                freq_centers[iteration_count + 1, mode] = np.dot(
                    spectral_domain[total_length // 2 :], np.abs(mode_fft_half) ** 2
                ) / np.sum(np.abs(mode_fft_half) ** 2)

        # Dual ascent step
        dual_vars += tau * (
            np.sum(mode_spectra[iteration_count + 1, :, :], axis=1) - positive_spectrum
        )

        # Check for convergence
        convergence_criteria = np.linalg.norm(
            mode_spectra[iteration_count + 1, :, :] - mode_spectra[iteration_count, :, :],
            ord="fro",
        ) / np.linalg.norm(mode_spectra[iteration_count, :, :], ord="fro")
        iteration_count += 1

    # Extract the final results
    max_iterations = iteration_count
    final_freq_centers = freq_centers[:max_iterations, :]

    # Symmetrize the spectrum and reconstruct the signal modes
    final_mode_spectra = np.zeros((total_length, num_modes), dtype=complex)
    final_mode_spectra[total_length // 2 :, :] = mode_spectra[
        max_iterations - 1, total_length // 2 :, :
    ]

    # Handling even/odd length symmetry
    if total_length % 2 == 0:
        final_mode_spectra[: total_length // 2, :] = np.conj(
            final_mode_spectra[-1 : total_length // 2 - 1 : -1, :]
        )
    else:
        final_mode_spectra[: total_length // 2, :] = np.conj(
            final_mode_spectra[-1 : total_length // 2 : -1, :]
        )

    # Inverse FFT to obtain the modes
    modes = np.real(np.fft.ifft(np.fft.ifftshift(final_mode_spectra, axes=0), axis=0))

    # Trim the modes to the original signal length
    modes = modes[total_length // 4 : 3 * total_length // 4, :]

    # Compute the final spectra for the modes
    mode_spectra_final = np.fft.fftshift(np.fft.fft(modes, axis=0), axes=0)

    return modes.T, mode_spectra_final, final_freq_centers


# Audio Paths
PATH_HC1 = (
    "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/raw/Carioca 1/Homens/HC1.wav"
)
PATH_HC1E = (
    "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/raw/Carioca 1/Homens/HC1e.wav"
)
PATH_HC1E_SEM_60 = "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/raw/Carioca 1/Homens/HC1e_sem_60.wav"
PATH_HC1_SEM_60 = "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/raw/Carioca 1/Homens/HC1_sem_60.wav"

# Constants sig, lowcut, highcut, fs, order
DOWNSAMPLE_FS = 400
BANDPASS_ORDER = 5
BNP_LOW = 99.5
BNP_HIGH = 100.5

ALPHA = 20_000  # Balancing parameter of the data-fidelity constraint
TAU = 0  # Noise-tolerance (no strict fidelity enforcement)
N_MODE = 1  # Number of modes to be recovered
DC = 0
TOL = 1e-7  # Tolerance of convergence criterion

F0 = 100
I_ = 1
EPSILON = 1e-7
TAU_RFA = int(1250)

N_DFT = 20_000
NUM_CYCLES = 160

pth = PATH_HC1E


## ENF Preprocessing and enhancement
# Read

sample_rate, data = wavfile.read(pth)

# Downsampling
down_sig, down_fs = downsample_ffmpeg(data, sample_rate, DOWNSAMPLE_FS)

# down_sig = cut_signal(down_sig, DOWNSAMPLE_FS, 10 * DOWNSAMPLE_FS, 13 * DOWNSAMPLE_FS)

# Bandpass
band_sig = bandpass_filter(down_sig, BNP_LOW, BNP_HIGH, DOWNSAMPLE_FS, BANDPASS_ORDER)

freqs = segmented_freq_estimation_DFT1(band_sig, DOWNSAMPLE_FS, NUM_CYCLES, N_DFT, F0)

plt.plot(freqs)
plt.xlabel("Cycles of nominal enf")
plt.ylabel("Frequency Hz")
plt.title("Frequency vs Time Raw")
plt.grid(True)
plt.show()


# VMD
for i in tqdm(range(1)):
    u_clean, _, _ = VMD(band_sig, ALPHA, TAU, N_MODE, DC, TOL)
    vmd_sig = u_clean[0]

freqs = segmented_freq_estimation_DFT1(vmd_sig, DOWNSAMPLE_FS, NUM_CYCLES, N_DFT, F0)
plt.plot(freqs)
plt.xlabel("Cycles of nominal enf")
plt.ylabel("Frequency Hz")
plt.title("Frequency vs Time VMD")
plt.grid(True)
plt.show()


# RFA
rfa_sig = RFA(vmd_sig, DOWNSAMPLE_FS, TAU_RFA, EPSILON, I_, F0)

freqs = segmented_freq_estimation_DFT1(rfa_sig, DOWNSAMPLE_FS, NUM_CYCLES, N_DFT, F0)

phases = segmented_phase_estimation_hilbert(rfa_sig, DOWNSAMPLE_FS, NUM_CYCLES, F0)
phases = np.unwrap(phases)
x = np.arange(len(phases))

# Uncut
plt.plot(freqs)
plt.xlabel("Cycles of nominal enf")
plt.ylabel("Frequency Hz")
plt.title("Frequency vs Time RFA")
plt.grid(True)
plt.show()

# Uncut
plt.plot(x, np.degrees(phases))
plt.xlabel("Cycles of nominal enf")
plt.ylabel("Phase (degrees)")
plt.title("Phase vs Time RFA")
plt.grid(True)
plt.show()
