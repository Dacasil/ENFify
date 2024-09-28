import numpy as np
from scipy.io import wavfile


# use enfify.examlpe_files
def _create_auth_tamp_clip(raw_sig, sample_rate, clip_length, max_cutlen, auth_path, tamp_path):
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
