"""Module for preprocessing the ENF signal."""

import math
import os
import random

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from scipy.stats import beta
from utils import read_wavfile

###.................Downsampling and bandpass filter.................###


def downsampling_python(s_raw, f_s, f_ds=1_000):
    """_summary_

    Args:
        s_raw (_type_): _description_
        f_s (_type_): _description_
        f_ds (_type_, optional): _description_. Defaults to 1_000.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if f_s % f_ds == 0:
        downsample_factor = f_s // f_ds
        s_ds = signal.decimate(s_raw, downsample_factor)
    else:
        nearest_downsample_factor = round(f_s / f_ds)
        new_sample_rate = f_s // nearest_downsample_factor

        if new_sample_rate == 0:
            raise ValueError(
                "Der berechnete Downsample-Faktor ist nicht sinnvoll. Überprüfen Sie die Eingabewerte."
            )

        s_ds = signal.decimate(s_raw, nearest_downsample_factor, ftype="fir")
        print("Not sufficient good implemented yet")
    return s_ds


def list_files_in_directory(input_dir, output_dir):
    """_summary_

    Args:
        input_dir (_type_): _description_
        output_dir (_type_): _description_

    Returns:
        _type_: _description_
    """
    # List all files in the directory
    files = os.listdir(input_dir)
    # Filter out directories, only keep files
    raw_files = [f for f in files if os.path.isfile(os.path.join(input_dir, f))]
    down_files = []
    files = []

    for raw in raw_files:
        down_file = output_dir + "/down_" + raw
        down_files.append(down_file)

    for raw in raw_files:
        files.append(input_dir + "/" + raw)

    return files, down_files


def downsampling(sig, fs, fs_down):
    """_summary_

    Args:
        sig (_type_): _description_
        fs (_type_): _description_
        fs_down (_type_): _description_

    Returns:
        _type_: _description_
    """
    in_file_path = "/tmp/tmp.wav"
    out_file_path = "/tmp/tmp_down.wav"
    wavfile.write(in_file_path, fs, sig)

    os.system(
        f". /home/$USER/miniforge3/etc/profile.d/conda.sh; conda activate enfify; ffmpeg -i {in_file_path} -ar {fs_down} {out_file_path}"
    )

    sig, fs = read_wavfile(out_file_path)
    os.remove(in_file_path)
    os.remove(out_file_path)
    return sig, fs


def bandpass_filter(sig, lowcut, highcut, fs, order):
    """_summary_

    Args:
        sig (_type_): _description_
        lowcut (_type_): _description_
        highcut (_type_): _description_
        fs (_type_): _description_
        order (_type_): _description_

    Returns:
        _type_: _description_
    """
    sos = signal.butter(order, [lowcut, highcut], btype="bandpass", output="sos", fs=fs)
    bandpass_sig = signal.sosfiltfilt(sos, sig)
    return bandpass_sig


###.................Generate tone and cut tone..................###


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


def cut_tones(sig, F_DS):
    """_summary_

    Args:
        sig (_type_): _description_
        F_DS (_type_): _description_

    Returns:
        _type_: _description_
    """

    m = len(sig)
    CUT_SAMPLES_LIMIT = 1 * F_DS

    cut_len = np.random.randint(0, CUT_SAMPLES_LIMIT)
    i_cut_start = np.random.randint(cut_len, m - cut_len)
    i_cut_end = i_cut_start + cut_len
    cut_sig = np.concatenate((sig[:i_cut_start], sig[i_cut_end:]))

    return cut_sig, i_cut_start, cut_len
