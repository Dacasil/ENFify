import numpy as np


def func_ENF_synthesis_corrupted_harmonic(
    fundamental_f, harmonic_index, corrupted_index, duration, fs, corrupt
):
    """
    Synthesizes a signal with corrupted harmonic frequencies based on a fundamental frequency

    Args:
        fundamental_f (float): The fundamental frequency
        harmonic_index (numpy.ndarray): Indices of the harmonic frequencies
        corrupted_index (numpy.ndarray): Indices of the harmonics to be corrupted
        duration (float): Duration of the signal in seconds
        fs (float): Sampling frequency
        corrupt (bool): Whether to add corruption to certain harmonics

    Returns:
        numpy.ndarray: The synthesized time-domain signal
        numpy.ndarray: The instantaneous frequencies for all harmonics
        numpy.ndarray: The individual harmonic signals
    """
    N = int(duration * fs)

    # Create fundamental IF
    f0 = np.random.randn(N)
    enf_freq = np.cumsum(f0) * 0.00005
    enf_freq = enf_freq / np.std(enf_freq) * np.sqrt(4.5e-4)
    enf_freq = enf_freq + fundamental_f
    enf_freqs = np.outer(harmonic_index, enf_freq)  # Instantaneous freqs across all harmonics

    if corrupt:
        index = np.intersect1d(harmonic_index, corrupted_index) - 1
        for i in range(len(index)):
            enf_freqs[index[i], :] = enf_freqs[index[i], :] + 5 * np.random.randn(N)

    # Instantaneous amplitudes and initial phases
    N_harmonic = len(harmonic_index)
    amps = 1 + np.random.randn(N_harmonic, N) * 0.005  # instantaneous amplitudes
    phases = np.random.uniform(0, 2 * np.pi, N_harmonic)  # initial phases

    # Synthesize time domain waveforms
    ENF_multi = np.zeros((N_harmonic, N))
    for n in range(N):
        ENF_multi[:, n] = amps[:, n] * np.cos(
            2 * np.pi / fs * np.sum(enf_freqs[:, : n + 1], axis=1) + phases
        )

    for i in range(min(6, N_harmonic)):
        ENF_multi[i, :] = ENF_multi[i, :] / np.linalg.norm(ENF_multi[i, :])

    sig = np.sum(ENF_multi, axis=0)
    sig_harmonics = ENF_multi
    sig = sig / np.linalg.norm(sig)  # ensure unit norm

    return sig, enf_freqs, sig_harmonics
