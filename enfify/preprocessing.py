"""Module for preprocessing the ENF signal."""

import os
import re
import tempfile
import math
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from loguru import logger
from scipy.io import wavfile
from scipy.signal import butter, decimate, lfilter, resample

# from enfify.utils import read_wavfile

# .................Downsampling and bandpass filter.................#


def downsample_scipy(sig, sample_rate, downsample_rate):
    """Downsample a numpy array using scipy, with antialiasing.

    Args:
        sig (numpy.ndarray): Raw signal.
        sample_rate (int or float): Current sampling frequency.
        downsample_rate (int or float): Target sampling frequency.

    Returns:
        numpy.ndarray: Downsampled signal.
        float: New sampling frequency.
    """

    if downsample_rate <= 0:
        raise ValueError("Target sampling rate must be greater than 0.")

    if downsample_rate >= sample_rate:
        logger.warning(
            "Not downsampling since the target sampling rate is greater than the current sampling rate."
        )
        return sig, sample_rate

    if sample_rate % downsample_rate == 0:
        # If the target sampling rate is an integer multiple of the original rate
        decimation_factor = int(sample_rate // downsample_rate)
        return decimate(sig, decimation_factor), downsample_rate  # Antialiasing is integrated here
    else:
        # Otherwise, use resampling
        # Log a warning about the need for an antialiasing filter
        logger.warning(
            "The target sampling rate is not an integer multiple of the current sampling rate. Resampling is used, which does not have an integrated antialiasing filter. Applying a lowpass filter."
        )

        # Apply a lowpass filter to ensure antialiasing
        cutoff_frequency = downsample_rate / 2.0
        filtered_sig = lowpass_filter(sig, cutoff_frequency, sample_rate)
        num_samples = int(len(filtered_sig) * downsample_rate / sample_rate)
        return resample(filtered_sig, num_samples), downsample_rate


def lowpass_filter(signal, cutoff, fs, order=5):
    """Apply a Butterworth lowpass filter for antialiasing.

    Args:
        signal (numpy.ndarray): Input signal.
        cutoff (float): Cutoff frequency of the filter (in Hz).
        fs (float): Sampling frequency of the signal (in Hz).
        order (int): Order of the filter.

    Returns:
        numpy.ndarray: Filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return lfilter(b, a, signal)


def downsample_ffmpeg(sig, sample_rate, downsample_rate):
    with (
        tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as input_file,
        tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as output_file,
    ):
        input_path = input_file.name
        output_path = output_file.name
        wavfile.write(input_path, sample_rate, sig)

        ffmpeg.input(input_path).output(
            output_path, ar=downsample_rate
        ).overwrite_output().global_args("-loglevel", "error").run()

        downsample_rate, downsampled_sig = wavfile.read(output_path)

        os.remove(input_path)
        os.remove(output_path)

    return downsampled_sig, downsample_rate


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


def cut_extract_number(file_name):
    """Extract number of Audio file anames to sort them

    Args:
        filenames (string)

    Returns:
        _type_: Paths + Names of cut and uncut audio files
    """

    match = re.match(r"(\d+)_cut_audio\.wav", file_name)
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


# .................Cut signal..................#


def cut_signal(sig, cut_start, cut_len):
    """Cuts numpy arrays

    Args:
        sig (nparray): numpy array signal
        F_DS (int or float): Sampling frequency of the signal

    Returns:
        _type_: Cut numpy array
    """
    cut_end = cut_start + cut_len
    cut_sig = np.concatenate((sig[:cut_start], sig[cut_end:]))

    return cut_sig


def cut_out_signal(sig, F_DS, cut_start, cut_len):
    """Cuts out numpy arrays

    Args:
        sig (nparray): numpy array signal
        F_DS (int or float): Sampling frequency of the signal

    Returns:
        _type_: Cut numpy array
    """

    cut_end = cut_start + cut_len
    cut_sig = sig[cut_start : cut_end + 1]

    return cut_sig


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


# Spatial and temporal features
def extract_temporal_features(psi_1_phases, fl, fn):
    """
    Function to extract temporal features T F fl × fn from phase sequence features ψ1.

    Parameters:
    - psi_1_list: A list of phase sequence features (arrays) for multiple audio files.
    - fl: The number of phase points contained in each frame (frame length).

    Returns:
    - temporal_features: A list of temporal feature matrices T F fl × fn for each audio file.
    """

    current_len = len(psi_1_phases)

    overlap = fl - math.floor(current_len / fn)

    # Split the phase sequence into frames using the calculated overlap
    frames = []
    for i in range(0, current_len - fl + 1, overlap):
        frame = psi_1_phases[i : i + fl]
        frames.append(frame)

    # Cases where the last frame is smaller than `fl`
    if len(psi_1_phases) % fl != 0:
        frame = psi_1_phases[-fl:]
        frames.append(frame)

    # Reshape the frames into a temporal feature matrix T F fl × fn
    feature_matrix = np.zeros((fl, fn))

    # Matrix
    for i in range(min(fn, len(frames))):
        feature_matrix[:, i] = frames[i]

    return feature_matrix


def extract_spatial_features(psi_1_phases, sn):
    """
    Function to extract spatial features S Fsn×sn from phase sequence features ψ1.

    Parameters:
    - psi_1_list: A list of phase sequence features (arrays) for multiple audio files.

    Returns:
    - spatial_features: A list of spatial feature matrices S Fsn×sn for each audio file.
    """

    ML = sn**2

    current_len = len(psi_1_phases)

    overlap = sn - math.ceil((ML - sn) / (current_len - sn))

    # Split the frame
    num_frames = (current_len - sn) // overlap + 1
    frames = []

    for i in range(0, num_frames * overlap, overlap):
        if i + sn <= current_len:
            frame = psi_1_phases[i : i + sn]
            frames.append(frame)
        else:
            break

    # Reshape into a spatial feature matrix (S Fsn×sn)
    feature_matrix = np.zeros((sn, sn))

    for i in range(min(sn, len(frames))):
        feature_matrix[i, : len(frames[i])] = frames[i]

    return feature_matrix
