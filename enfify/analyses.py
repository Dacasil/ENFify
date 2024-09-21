"""Module for analyses / classification of the features to finish the
tampering detection."""

import numpy as np
from scipy.stats import zscore
from sklearn.metrics import roc_curve


def feature_zscore(phases, trunclen=10):
    """Noch nicht ausgereifte Funktion, die in authentic/tampered klassifizieren soll.
    Parameter Gamma muss trainiert werden.
    Ggf. ist eine Mittelung der Z-Scores sinnvoll."""

    derivative = np.diff(phases)
    z_scores = np.abs(zscore(derivative))[trunclen:-trunclen]

    return np.max(z_scores)


def classification_dacasil(phases, x):
    # fast wie find_cut_in_phases, aber bool output
    """
    Classify phase data based on detected discontinuities.

    Args:
        phases (np.ndarray): Array of phase values
        x (np.ndarray): Corresponding x values for the phases

    Returns:
        bool: True if no significant discontinuities are found, False otherwise
    """

    range_threshold = 20
    window_size = 10
    second_der = np.gradient(np.gradient(phases, x), x)

    z_scores = np.abs(zscore(second_der))
    outliers = np.array(np.where(z_scores > 3))

    if not np.any(outliers):
        return True

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
            return True

        start = discontinuities[0][0]
        end = discontinuities[0][1]

        # phases_new = phases[int(start) - 200 : int(end) + 200]
        # x_new = x[int(start) - 200 : int(end) + 200]

        return False


def find_cut_in_phases(phases, x):
    """
    Identify discontinuities in phase data based on second derivatives.

    Args:
        phases (np.ndarray): Array of phase values
        x (np.ndarray): Corresponding x values for the phases

    Returns:
        tuple: A tuple containing:
            - phases_new (list): List of phase segments around discontinuities
            - x_new (list): List of corresponding x segments
            - discontinuities (np.ndarray): Indices of identified discontinuities
    """

    range_threshold = 20
    window_size = 10
    second_der = np.gradient(np.gradient(phases, x), x)

    z_scores = np.abs(zscore(second_der))
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


def feature_rodriguez(phases):
    """Calculate the Rodriguez feature based on estimated phases.

    Args:
        phases (np.ndarray[float]): Array of estimated phases

    Returns:
        float: Feature value
    """
    phi_diff = np.diff(phases)
    m_phi_diff = np.mean(phi_diff)
    F = 100 * np.log(np.mean((phi_diff - m_phi_diff) ** 2))
    return F


def classification_rodriguez(feature, gamma):
    return feature < gamma


def eer_feature_threshold(labels, features):
    """Calculate the threshold of the feature for the equal error rate.

    Args:
        labels (numpy.ndarray): Ground truth binary labels (0 or 1)
        features (numpy.ndarray): Feature scores or probabilities associated with the positive class

    Returns:
        float: The threshold value that corresponds to the equal error rate
    """
    fpr, tpr, thresholds = roc_curve(labels, features)
    eer_threshold = thresholds[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    return eer_threshold


if __name__ == "__main__":
    pass
