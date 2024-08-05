import numpy as np
#import scipy.signal as signal
from tqdm import tqdm
import math
import cmath
from scipy.signal import get_window
from scipy.fft import fft
import sys
sys.path.insert(0, '../sources')
from ENF_frequency_phase_estimation import phase_estimation_DFT0, phase_estimation_DFT1, segmented_freq_estimation_DFT1

###...........................RFA................................###
def z_SFM(sig, n , fs, alpha, tau):
    sum_sig = 0
    sig_padded = np.pad(sig, (tau, tau), 'constant')

    # n+tau
    for i in range(n+tau+1):
        sum_sig += sig_padded[i]
    z = cmath.exp(1j*2*math.pi*(1/fs)*alpha*sum_sig)
    return z


def z_SFM_complex(sig, n, fs, alpha, tau):
    sum_sig = 0
    sig_padded = np.pad(sig, (tau, tau), 'constant')

    # n-tau
    for i in range(n-tau+1):
        sum_sig += sig_padded[i]
    z = cmath.exp(-1j*2*math.pi*(1/fs)*alpha*sum_sig)
    return z


def kernel_function(sig, f, n, fs, alpha, tau):
    tau_dash = int(tau + np.round(fs/(4*f)))
    auto_corr = z_SFM(sig, n, fs, alpha, tau) * z_SFM_complex(sig, n, fs, alpha, tau)
    auto_corr_dash = z_SFM(sig, n, fs, alpha, tau_dash) * z_SFM_complex(sig, n, fs, alpha, tau_dash)
    
    Kernel = (auto_corr**((1/fs)*f*tau*math.pi*np.sin(2*math.pi*(1/fs)*f*tau)) 
              * auto_corr_dash**((1/fs)*f*tau*math.pi*np.cos(2*math.pi*(1/fs)*f*tau)))
    return Kernel


def RFA(sig, fs, tau, epsilon, I, estimated_enf):
    print("test")
    # Initialise
    k = 1
    Nx = len(sig)
    alpha = 1/4 * fs/np.max(sig)
    f_start = estimated_enf * np.ones(Nx)
    
    while k <= I:
        denoised_sig = []

        for n in tqdm(range(0, Nx-1)):
            f = f_start[n]
            phase_of_kernel = 0
            
            for m in range(1,tau+1):
                phase_of_kernel += np.angle(kernel_function(sig, f, n, fs, alpha, m))

            denoised_sig.append(phase_of_kernel/((tau+1)*tau*math.pi*alpha))
        print(denoised_sig)
        peak_freqs = segmented_freq_estimation_DFT1(denoised_sig, fs, num_cycles = 5, N_DFT= 20_000, nominal_enf=estimated_enf)
        sig_len = int(len(sig)/len(peak_freqs))
        new_freqs = np.ones(len(denoised_sig))

        for l in range(len(peak_freqs)):
            new_freqs[l*sig_len:(2*sig_len + 2*l*sig_len)] = peak_freqs[l]

        f = f_start
        numerator = 0
        denominator = 0
        print(new_freqs)

        for s in range(len(new_freqs)):
            numerator += (new_freqs[s]-f[s])**2
            denominator += (f[s])**2
        
        if numerator/denominator <= epsilon:
            return denoised_sig

        f_start = new_freqs
        sig = denoised_sig
        denoised_signal = denoised_sig
        k+=1
    return denoised_signal


###...................................Variational Mode Decomposition...................................###

def VariationalModeDecomposition(signal, alpha, tau, num_modes, enforce_DC, tolerance):
    
    # Mirror signal at boundaries
    if len(signal) % 2:
        midpoint = math.ceil(len(signal) / 2)
        left_mirror = np.concatenate((np.flipud(signal[:midpoint-1]), signal), axis=0)
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
    positive_spectrum[:total_length // 2] = 0

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
    while (convergence_criteria > tolerance and iteration_count < max_iterations - 1):
        mode_sum = mode_spectra[iteration_count, :, num_modes - 1] + mode_sum - mode_spectra[iteration_count, :, 0]
        
        mode_spectra[iteration_count + 1, :, 0] = (positive_spectrum - mode_sum - dual_vars[iteration_count, :] / 2) / \
                                                  (1 + mode_alphas[0] * (spectral_domain - freq_centers[iteration_count, 0]) ** 2)
        
        if not enforce_DC:
            freq_centers[iteration_count + 1, 0] = np.dot(spectral_domain[total_length // 2:],
                                                          abs(mode_spectra[iteration_count + 1, total_length // 2:, 0]) ** 2) / \
                                                   np.sum(abs(mode_spectra[iteration_count + 1, total_length // 2:, 0]) ** 2)

        for mode in range(1, num_modes):
            mode_sum = mode_spectra[iteration_count + 1, :, mode - 1] + mode_sum - mode_spectra[iteration_count, :, mode]
            mode_spectra[iteration_count + 1, :, mode] = (positive_spectrum - mode_sum - dual_vars[iteration_count, :] / 2) / \
                                                         (1 + mode_alphas[mode] * (spectral_domain - freq_centers[iteration_count, mode]) ** 2)
            freq_centers[iteration_count + 1, mode] = np.dot(spectral_domain[total_length // 2:],
                                                             abs(mode_spectra[iteration_count + 1, total_length // 2:, mode]) ** 2) / \
                                                      np.sum(abs(mode_spectra[iteration_count + 1, total_length // 2:, mode]) ** 2)

        dual_vars[iteration_count + 1, :] = dual_vars[iteration_count, :] + tau * (np.sum(mode_spectra[iteration_count + 1, :, :], axis=1) - positive_spectrum)
        
        iteration_count += 1
        
        convergence_criteria = np.spacing(1)
        for mode in range(num_modes):
            convergence_criteria += (1 / total_length) * np.dot((mode_spectra[iteration_count, :, mode] - mode_spectra[iteration_count - 1, :, mode]),
                                                               np.conj(mode_spectra[iteration_count, :, mode] - mode_spectra[iteration_count - 1, :, mode]))

        convergence_criteria = np.abs(convergence_criteria)

    # Postprocessing to extract modes and their spectra
    max_iterations = min(max_iterations, iteration_count)
    final_freq_centers = freq_centers[:max_iterations, :]
    
    half_idxs = np.flip(np.arange(1, total_length // 2 + 1), axis=0)
    final_mode_spectra = np.zeros((total_length, num_modes), dtype=complex)
    final_mode_spectra[total_length // 2:total_length, :] = mode_spectra[max_iterations - 1, total_length // 2:total_length, :]
    final_mode_spectra[half_idxs, :] = np.conj(mode_spectra[max_iterations - 1, total_length // 2:total_length, :])
    final_mode_spectra[0, :] = np.conj(final_mode_spectra[-1, :])

    modes = np.zeros((num_modes, len(time_domain)))
    for mode in range(num_modes):
        modes[mode, :] = np.real(np.fft.ifft(np.fft.ifftshift(final_mode_spectra[:, mode])))
    
    modes = modes[:, total_length // 4: 3 * total_length // 4]

    mode_spectra_final = np.zeros((modes.shape[1], num_modes), dtype=complex)
    for mode in range(num_modes):
        mode_spectra_final[:, mode] = np.fft.fftshift(np.fft.fft(modes[mode, :]))

    return modes, mode_spectra_final, final_freq_centers