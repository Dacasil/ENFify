import numpy as np
from scipy.io import wavfile


def sym_phase_interval(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def add_defaults(config, defaults):
    """
    Recursively add default values to a config dictionary.

    Args:
        config (dict): The configuration dictionary to update
        defaults (dict): The dictionary containing default values
    """
    for key, value in defaults.items():
        if key not in config or config[key] is None:
            config[key] = value
        elif isinstance(value, dict) and isinstance(config.get(key), dict):
            add_defaults(config[key], value)


def read_wavfile(file_path):
    """
    Reads a WAV file and processes the audio data.

    Args:
        file_path (str): Path to the WAV file

    Returns:
        numpy.ndarray: The audio data.
        int: The sampling rate of the audio file

    Raises:
        Exception: If the file has an unusual number of channels or an unexpected data structur
    """
    # Read the WAV file
    fs, data = wavfile.read(file_path)

    # Check the number of channels
    if len(data.shape) == 1:
        return data, fs

    elif len(data.shape) == 2:
        channels = data.shape[1]
        if channels == 2:
            data = np.mean(data, axis=1)
            return data, fs
        else:
            raise Exception(f"The file has {channels} channels, which is unusual.")
    else:
        raise Exception("The structure of the audio data is unexpected.")
