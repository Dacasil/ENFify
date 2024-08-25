"""Module for preprocessing the ENF signal."""

import os
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from .utils import read_wavfile

# .................Downsampling and bandpass filter.................#


def downsampling_python(s_raw, f_s, f_ds=1_000):
    """Downsampling of numpy array via python

    Args:
        s_raw (Numpy array):raw signal
        f_s (int or float): current sampling freq
        f_ds (int or float, optional): downsampling sampling freq

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


# def downsampling_alpha(in_file_path, out_file_path, fs_down):
#     """Downsample a audio signal with ffmpeg to reduce numrical costs and enable signale preprocessing and denoising

#     Args:
#         in_file_path (str): Where data to downsample is stored + filename
#         out_file_path (str): Where downsampled data to store + filename

#     Returns:
#         None
#     """

#     if platform.system() == "Windows":
#         # Windows-specific Command
#         conda_activate = (
#             f"call {os.getenv('USERPROFILE')}\\miniforge3\\Scripts\\activate.bat enfify"
#         )
#         ffmpeg_command = f"ffmpeg -i {in_file_path} -ar {fs_down} {out_file_path}"
#         os.system(f"{conda_activate} && {ffmpeg_command}")

#     else:
#         # Unix-specific Command (Linux, macOS)
#         conda_activate = ". /home/$USER/miniforge3/etc/profile.d/conda.sh; conda activate enfify"
#         ffmpeg_command = f"ffmpeg -i {in_file_path} -ar {fs_down} {out_file_path}"
#         os.system(f"{conda_activate} && {ffmpeg_command}")


def downsampling_alpha(sig, fs, fs_down):
    """
    Downsampling for the alpha Version

    Args:
        sig (): _description_
        fs (_type_): _description_
        fs_down (_type_): _description_

    Returns:
        _type_: _description_
    """
    in_file_path = "tmp.wav"  # maybe move to /tmp but not generic for windows
    out_file_path = "tmp_down.wav"

    wavfile.write(in_file_path, fs, sig)

    # os.system(
    #     f". /home/$USER/miniforge3/etc/profile.d/conda.sh; conda activate enfify; ffmpeg -i {in_file_path} -ar {fs_down} {out_file_path}"
    # )
    subprocess.run(
        ["ffmpeg", "-i", in_file_path, "-ar", str(fs_down), out_file_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    sig, fs = read_wavfile(out_file_path)
    os.remove(in_file_path)
    os.remove(out_file_path)
    return sig, fs


def bandpass_filter(sig, lowcut, highcut, fs, order):
    """Bandpass Filter to cut out unwanted frequencys of a signal

    Args:
        sig (numpy array_):Signal to filter
        lowcut (int or float): Lower limit of the wanted frequency
        highcut (int or float): Upper limit of the wanted frequency
        fs (int or float): Sampling frequency
        order (int or float): Precision order of the bandpass filter

    Returns:
        Numpy Array: BAndpassed signal
    """

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype="band", output="sos")
    bandpass_sig = signal.sosfiltfilt(sos, sig)

    return bandpass_sig


def butter_bandpass_test(lowcut, highcut, fs, order=5):
    """Test the butter bandpass filter

    Args:
        lowcut (int or float): Lower limit of the wanted frequency
        highcut (int or float): Upper limit of the wanted frequency
        fs (int or float): Sampling frequency
        order (int or float): Precision order of the bandpass filter

    Returns:
        Numpy Array: Bandpass Filter
    """

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype="band", output="sos")
    w, h = signal.sosfreqz(sos, worN=20000)
    plt.semilogx((fs * 0.5 / np.pi) * w, abs(h))
    return sos


# .................Extract and generate File names..................#


def extract_number(file_name):
    """Extract number of Audio file anames to sort them

    Args:
        filenames (string)

    Returns:
        _type_: Paths + Names of cut and uncut audio files
    """

    match = re.match(r"(\d+)_audio\.wav", file_name)
    return int(match.group(1)) if match else float("inf")


def list_files_in_directory(input_dir, output_dir):
    """File names and directory paths for donwsampling audio files with ffmpeg

    Args:
        input_dir (string): directory names where the audio files are stored
        output_dir (string): directory names where the new downsampled audio files should be stored

    Returns:
        _type_: Paths + Names of downsampled and original audio files
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


def ffmpeg_filenames_cut(input_dir, output_dir):
    """File names and directory paths for cut and uncut samples with ffmpeg

    Args:
        input_dir (str): Directory of the input audio signals
        output_dir (str): Directory where to save the the cut audio signals

    Returns:
        _type_: Paths + Names of cut and uncut audio files
    """

    try:
        # List all files in the directory
        files = os.listdir(input_dir)
        files.sort(key=extract_number)
        # Filter out directories, only keep files
        raw_files = [f for f in files if os.path.isfile(os.path.join(input_dir, f))]
        cut_files = []
        files = []

        for raw in raw_files:
            cut_file = output_dir + "/cut_" + raw
            cut_files.append(cut_file)

        for raw in raw_files:
            files.append(input_dir + "/" + raw)

        return files, cut_files

    except FileNotFoundError:
        return f"The directory {input_dir} does not exist."
    except PermissionError:
        return f"Permission denied to access {input_dir}."


# .................Cut signbal..................#


def cut_tones(sig, F_DS):
    """Random cuts numpy arrays

    Args:
        sig (nparray): numpy array signal
        F_DS (int or float): Sampling frequency of the signal

    Returns:
        _type_: Cut numpy array
    """

    m = len(sig)
    CUT_SAMPLES_LIMIT = 1 * F_DS

    cut_len = np.random.randint(0, CUT_SAMPLES_LIMIT)
    i_cut_start = np.random.randint(cut_len, m - cut_len)
    i_cut_end = i_cut_start + cut_len
    cut_sig = np.concatenate((sig[:i_cut_start], sig[i_cut_end:]))

    return cut_sig, i_cut_start, cut_len


def cut_audio(input_file, output_file, cut_begin, cut_len):
    """
    Single cut in audio files with ffmpeg (.wav or .mp3 files)

    Args:
        input_file (string): Audio to cut
        output_file (string): where to save the cut audio
        cut_begin (int or float): begin of the cut
        cut_len (int or float): len of the cut out audio part

    Returns:
        _type_: Returns nothing, cut audio file gets saved in output_file path
    """

    # Ensure paths with spaces are quoted
    input_file_quoted = input_file.replace(" ", r"\ ")
    output_file_quoted = output_file.replace(" ", r"\ ")

    # Command to execute
    command = (
        f". /home/$USER/miniforge3/etc/profile.d/conda.sh; "
        f"conda activate enfify; "
        f"ffmpeg -i {input_file_quoted} -filter_complex "
        f'"[0]atrim=end={cut_begin},asetpts=PTS-STARTPTS[a1]; '
        f"[0]atrim=start={cut_len},asetpts=PTS-STARTPTS[a2]; "
        f'[a1][a2]concat=n=2:v=0:a=1[out]" '
        f'-map "[out]" {output_file_quoted}'
    )

    # Execute the command
    os.system(command)


def mult_cut_audio(
    input_file, output_file, cut_begin_1, cut_end_1, cut_begin_2, cut_end_2, cut_begin_3, cut_end_3
):
    """
    Three Cuts in audio files with ffmpeg (.wav or .mp3 files)

    Args:
        input_file (string): Audio to cut
        output_file (string): where to save the cut audio
        cut_begin_1-3 (int or float): begin of the cuts
        cut_end_1-3 (int or float): end of the cuts

    Returns:
        _type_: Returns nothing, cut audio file gets saved in output_file path
    """

    # Ensure paths with spaces are quoted
    input_file_quoted = input_file.replace(" ", r"\ ")
    output_file_quoted = output_file.replace(" ", r"\ ")

    # Command to execute
    command = (
        f". /home/$USER/miniforge3/etc/profile.d/conda.sh; "
        f"conda activate enfify; "
        f"ffmpeg -i {input_file_quoted} -filter_complex "
        f'"[0]atrim=end={cut_begin_1},asetpts=PTS-STARTPTS[a1]; '
        f"[0]atrim=start={cut_end_1}:end={cut_begin_2},asetpts=PTS-STARTPTS[a2]; "
        f"[0]atrim=start={cut_end_2}:end={cut_begin_3},asetpts=PTS-STARTPTS[a3]; "
        f"[0]atrim=start={cut_end_3},asetpts=PTS-STARTPTS[a4]; "
        f'[a1][a2][a3][a4]concat=n=4:v=0:a=1[out]" '
        f'-map "[out]" {output_file_quoted}'
    )

    # Execute the command
    os.system(command)
