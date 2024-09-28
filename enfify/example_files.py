import numpy as np
from scipy.io import wavfile


def func_ENF_synthesis_corrupted_harmonic(
    fundamental_f=50,
    harmonic_index=range(1, 7),
    corrupted_index=range(3, 7),
    duration=12,
    fs=1000,
    corrupt=False,
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
        float: The sampling frequency
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
        enf_freqs[index, :] += 5 * np.random.randn(len(index), N)

    # Instantaneous amplitudes and initial phases
    N_harmonic = len(harmonic_index)
    amps = 1 + np.random.randn(N_harmonic, N) * 0.005  # instantaneous amplitudes
    phases = np.random.uniform(0, 2 * np.pi, N_harmonic)  # initial phases

    # Synthesize time domain waveforms using broadcasting and cumulative sum
    ENF_multi = amps * np.cos(
        2 * np.pi / fs * np.cumsum(enf_freqs, axis=1) + phases[:, np.newaxis]
    )

    # Normalize harmonics and sum them
    for i in range(min(6, N_harmonic)):
        ENF_multi[i, :] /= np.linalg.norm(ENF_multi[i, :])

    sig = np.sum(ENF_multi, axis=0)
    sig /= np.linalg.norm(sig)  # ensure unit norm

    return sig, fs


def create_auth_tamp_clip(raw_sig, sample_rate, clip_length, max_cutlen, auth_path, tamp_path):
    """
    Creates an authenticated and a tampered audio clip from the given raw signal.

    Args:
        raw_sig (ndarray): The raw audio signal.
        sample_rate (int): The sample rate of the audio signal.
        clip_length (float): The desired length of the audio clip in seconds.
        max_cutlen (float): The maximum length of the cut in seconds.
        auth_path (Path): The path of the authenticated audio clip.
        tamp_path (Path): The path of the tampered audio clip.

    Returns:
        int: The start index of the cut in the raw signal.
        int: The length of the cut in the raw signal.
    """
    cliplen_samples = int(clip_length * sample_rate)

    auth_sig = raw_sig[:cliplen_samples]
    wavfile.write(auth_path, sample_rate, auth_sig)

    max_cutlen_samples = int(max_cutlen * sample_rate)
    cutlen_samples = np.random.randint(max_cutlen_samples) + 1
    start = np.random.randint(0, cliplen_samples - cutlen_samples)
    tamp_sig = np.delete(raw_sig.copy(), slice(start, start + cutlen_samples))[:cliplen_samples]
    wavfile.write(tamp_path, sample_rate, tamp_sig)

    return start, cutlen_samples
