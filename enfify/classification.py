from math import ceil

import numpy as np


def sectioning(array, section_len, min_overlap):
    """Split an array into sections of a given length with a minimum overlap so that the whole array is covered.

    Args:
        array (numpy.ndarray): Array to split.
        section_len (int): Length of each section.
        min_overlap (int): Minimum overlap between sections.

    Returns:
        List[numpy.ndarray]: List of sections.
    """
    start = np.linspace(
        0, len(array), ceil(len(array) / (section_len - min_overlap)), endpoint=False, dtype=int
    )
    end = start + section_len
    end[-1] = len(array)
    start[-1] = len(array) - section_len
    return [array[s:e] for s, e in zip(start, end)]


def cnn_classifier(*args, **kwargs):
    raise NotImplementedError
