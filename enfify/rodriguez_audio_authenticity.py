"""Methods for the Audio Authenticity algorithm from the Rodriguez Paper."""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve

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

    range_threshold = 20
    window_size = 10
    second_der = np.gradient(np.gradient(phases, x), x)

    z_scores = np.abs(stats.zscore(second_der))
    outliers = np.array(np.where(z_scores > 5))

    if not np.any(outliers):
        return phases, x, outliers

    else:
        discontinuities = []
        i = 0

        while i < len(outliers[0]) - 1:
            start = outliers[0][i]
            while (
                i < len(outliers[0]) - 1
                and (outliers[0][i + 1] - outliers[0][i]) <= range_threshold
            ):
                i += 1
            end = outliers[0][i]

            # Search for the cut discontinuitites
            if end - start >= window_size:
                segment = second_der[start : end + 1]
                pos_count = np.sum(segment > 0)
                neg_count = np.sum(segment < 0)

                if pos_count > 0 and neg_count > 0:
                    discontinuities.append((start, end))

            i += 1

        discontinuities = np.array(discontinuities)

        if not np.any(discontinuities):
            return phases, x, discontinuities

        phases_new = []
        x_new = []

        for i in range(len(discontinuities)):
            start = discontinuities[i][0]
            end = discontinuities[i][1]

            phases_new.append(phases[int(start) - 200 : int(end) + 200])
            x_new.append(x[int(start) - 200 : int(end) + 200])

        return phases_new, x_new, discontinuities
