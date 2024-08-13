"""Methods for the Audio Authenticity algorithm from the Rodriguez Paper."""

import numpy as np
from sklearn.metrics import roc_curve
from scipy import stats


# ..........................Feature Estimation.........................#


def feature(phases):
    """_summary_

    Args:
        phases (_type_): _description_

    Returns:
        _type_: _description_
    """
    phi_diff = np.diff(phases)
    m_phi_diff = np.mean(phi_diff)
    F = 100 * np.log(np.mean((phi_diff - m_phi_diff) ** 2))
    return F


# ..........................Lambda.........................#


def Lambda(uncut_F, cut_F):
    """_summary_

    Args:
        uncut_F (_type_): _description_
        cut_F (_type_): _description_

    Returns:
        _type_: _description_
    """
    num_samples = len(uncut_F)
    labels = np.concatenate([np.zeros(num_samples), np.ones(num_samples)])
    features = np.concatenate([uncut_F, cut_F])

    # Calculate lambda
    fpr, tpr, thresholds = roc_curve(labels, features)
    eer_threshold = thresholds[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    return eer_threshold


def lambda_accuracy(uncut_features, cut_features, Lambda):
    """_summary_

    Args:
        uncut_features (_type_): _description_
        cut_features (_type_): _description_
        Lambda (_type_): _description_

    Returns:
        _type_: _description_
    """

    n_cut = len(cut_features)
    n_uncut = len(uncut_features)

    p_cut = np.sum(cut_features >= Lambda) / n_cut
    p_uncut = np.sum(uncut_features < Lambda) / n_uncut

    p_characterization = (p_cut * n_cut + p_uncut * n_uncut) / (n_cut + n_uncut)

    return p_characterization


def find_cut_in_phases(phases, x):
    """_summary_

    Args:
        phases (_type_): _description_
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    second_der = np.gradient(np.gradient(phases, x), x)

    z_scores = np.abs(stats.zscore(second_der))
    ausreisser = np.array(np.where(z_scores > 10))

    if np.any(ausreisser) == False:
        return phases, x, ausreisser
    
    else: 
        phases_new = phases[int(np.min(ausreisser)) - 200 : int(np.max(ausreisser)) + 200]
        x_new = x[int(np.min(ausreisser)) - 200 : int(np.max(ausreisser)) + 200]

        return phases_new, x_new, ausreisser
