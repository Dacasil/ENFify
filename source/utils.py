import numpy as np


def sym_phase_interval(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi
