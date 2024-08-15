import math
import random

import numpy as np
from scipy.stats import beta
from tqdm import tqdm


def generate_random_number(min_value, max_value, decimal_places):
    """_summary_

    Args:
        min_value (_type_): _description_
        max_value (_type_): _description_
        decimal_places (_type_): _description_

    Returns:
        _type_: _description_
    """
    number = random.uniform(min_value, max_value)
    formatted_number = round(number, decimal_places)
    return formatted_number


def generate_s_tone(Fs, f0, phi0, M):
    """_summary_

    Args:
        Fs (_type_): _description_
        f0 (_type_): _description_
        phi0 (_type_): _description_
        M (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = np.arange(M)
    s_tone = np.cos((2 * np.pi * f0 * n) / Fs + phi0)
    return n, s_tone


def create_tones(Fs, M, toneSample_num):
    """_summary_

    Args:
        Fs (_type_): _description_
        M (_type_): _description_
        toneSample_num (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Generate random sinus waves
    lower_bound_freq = 59.0  # in Hz
    upper_bound_freq = 61.0

    frequencies = []
    for i in range(toneSample_num):
        frequencies.append(generate_random_number(lower_bound_freq, upper_bound_freq, 1))
    frequencies = np.array(frequencies)

    phases = []
    for i in range(toneSample_num):
        phases.append(generate_random_number(0, np.pi / 2, 1))
    phases = np.array(phases)

    s_tones = []
    for i in range(len(frequencies)):
        n, s_tone = generate_s_tone(Fs, frequencies[i], phases[i], M)
        s_tones.append(s_tone)
    tones = np.array(s_tones)

    cut_tones = []
    cut_coords = []
    cut_len = []

    for i in range(len(frequencies)):
        period_length = math.floor(Fs / frequencies[i])
        start_num = int(
            generate_random_number(100, M - period_length - 100, 0)
        )  # Random number between 0 and N - (period length)
        end_num = generate_random_number(
            0, (period_length - 1), 0
        )  # Random number between 1 and period_length-1)
        end_num = int(start_num + end_num)
        cut_coords.append(start_num)
        cut_tones.append(np.concatenate((tones[i][:start_num], tones[i][end_num:])))
        cut_len.append(np.degrees(2 * np.pi * (end_num - start_num) / period_length))

    return frequencies, phases, tones, cut_tones, cut_coords, cut_len


def random_signal(AMPLITUDE, DURATION, F_DS, NOMINAL_ENF, PM_NOMINAL, CUT_SAMPLES_LIMIT):
    """_summary_

    Args:
        AMPLITUDE (_type_): _description_
        DURATION (_type_): _description_
        F_DS (_type_): _description_
        NOMINAL_ENF (_type_): _description_
        PM_NOMINAL (_type_): _description_
        CUT_SAMPLES_LIMIT (_type_): _description_

    Returns:
        _type_: _description_
    """

    m = DURATION * F_DS

    # random generation
    f_tone = NOMINAL_ENF - PM_NOMINAL + 2 * PM_NOMINAL * beta.rvs(2, 2)
    phi_0 = np.random.uniform(0, 2 * np.pi)

    i_cut = np.random.randint(0, m)
    cut_len = np.random.randint(0, CUT_SAMPLES_LIMIT)  # in samples

    n_cut = np.arange(m + cut_len)
    n_cut = np.delete(n_cut, slice(i_cut, i_cut + cut_len))

    n_uncut = np.arange(m)

    sig_cut = AMPLITUDE * np.cos((2 * np.pi * f_tone) * (n_cut / F_DS) + phi_0)
    sig_uncut = AMPLITUDE * np.cos((2 * np.pi * f_tone) * (n_uncut / F_DS) + phi_0)

    return sig_uncut, sig_cut


def func_ENF_synthesis_corrupted_harmonic(
    fundamental_f, harmonic_index, corrupted_index, duration, fs, corrupt
):
    N = int(duration * fs)

    # Create fundamental IF
    f0 = np.random.randn(N)
    enf_freq = np.cumsum(f0) * 0.0005
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
    for n in tqdm(range(N)):
        ENF_multi[:, n] = amps[:, n] * np.cos(
            2 * np.pi / fs * np.sum(enf_freqs[:, : n + 1], axis=1) + phases
        )

    for i in range(min(6, N_harmonic)):
        ENF_multi[i, :] = ENF_multi[i, :] / np.linalg.norm(ENF_multi[i, :])

    sig = np.sum(ENF_multi, axis=0)
    sig_harmonics = ENF_multi
    sig = sig / np.linalg.norm(sig)  # ensure unit norm

    return sig, enf_freq, sig_harmonics
