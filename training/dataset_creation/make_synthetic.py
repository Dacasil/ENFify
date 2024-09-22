import json
import os
import sys

import numpy as np
from locallib import create_auth_tamp_clip
from tqdm import tqdm

from enfify import INTERIM_DATA_DIR

NUM_CLIPS = 2000  # Number of files to generate

CLIP_LENGTH = 10  # in seconds
MAX_CUTLEN = 2  # in seconds

SAMPLE_RATE = 1000  # Sampling rate in Hz
NOMINAL_FREQ = 50  # Nominal frequency in Hz


def func_ENF_synthesis_corrupted_harmonic(
    fundamental_f=NOMINAL_FREQ,
    harmonic_index=range(1, 7),
    corrupted_index=range(3, 7),
    duration=CLIP_LENGTH + MAX_CUTLEN,
    fs=SAMPLE_RATE,
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
    sig_harmonics = ENF_multi  # noqa: F841
    sig = sig / np.linalg.norm(sig)  # ensure unit norm

    return sig, fs


if __name__ == "__main__":
    interim_dir = INTERIM_DATA_DIR / "Synthetic"
    interim_dir.mkdir(parents=True)

    cliplen_samples = int(CLIP_LENGTH * SAMPLE_RATE)

    cut_info = {}
    for i in tqdm(range(NUM_CLIPS)):
        auth_path = interim_dir / f"synth-{i:04}-auth.wav"
        tamp_path = interim_dir / f"synth-{i:04}-tamp.wav"

        # if auth_path.exists() and tamp_path.exists():
        #     continue

        raw_sig, _ = func_ENF_synthesis_corrupted_harmonic()

        start_ind, cutlen_samples = create_auth_tamp_clip(
            raw_sig, SAMPLE_RATE, CLIP_LENGTH, MAX_CUTLEN, auth_path, tamp_path
        )

        cut_info[tamp_path.name] = {"start": start_ind, "cutlen": cutlen_samples}

    cut_info_path = interim_dir / "cut_info.json"
    with open(cut_info_path, "w") as f:
        json.dump(cut_info, f, indent=4)
