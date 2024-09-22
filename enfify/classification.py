from math import ceil

import numpy as np


def sectioning(array, section_len, min_overlap):
    """Split an array into sections of a given length with a minimum overlap so that the whole array is covered.
    Used e.g. for mapping out a feature array for a CNN with a certain input length.

    Args:
        array (numpy.ndarray): Array to split.
        section_len (int): Length of each section.
        min_overlap (int): Minimum overlap between sections.

    Returns:
        List[numpy.ndarray]: List of sections.
    """
    start = np.linspace(
        0,
        len(array) - section_len,
        ceil(len(array) / (section_len - min_overlap)),
        dtype=int,
    )
    end = start + section_len
    return [array[s:e] for s, e in zip(start, end)]


def cnn_classifier(*args, **kwargs):
    raise NotImplementedError
