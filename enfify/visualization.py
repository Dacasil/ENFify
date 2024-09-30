from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore


def report(prediction, confidence, audio_file, times, feature_freq_vector, feature_phase_vector):
    # Compute variables
    outpath = Path(".") / f"report_{audio_file.stem}.pdf"

    # Init
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69))

    # Plotting
    ax1.plot(times, feature_freq_vector, label="Frequency", color="tab:blue")
    ax2.plot(times, np.degrees(feature_phase_vector), label="Phase", color="tab:orange")
    # if prediction:  # TODO
    #     discontinuities = find_cut_in_phases(feature_phase_vector, times)
    #     for i, (start, end) in enumerate(discontinuities.T):
    #         ax2.axvspan(times[start], times[end], color="tab:red", alpha=0.2)

    # Limits
    # ax1.set_ylim(lowcut, highcut)

    # Labels
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_title("Frequency Estimation")
    ax1.grid()
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Phase [°]")
    ax2.set_title("Phase Estimation")
    ax2.grid()
    label = "Tampered" if prediction else "Authentic"
    fig.suptitle(f"Report for {audio_file.name} - {label} ({confidence*100:.1f}% confidence)")

    # Legend
    ax1.legend()
    ax2.legend()

    # Save
    plt.savefig(outpath, dpi=300)


def find_cut_in_phases(phases, x):
    """
    Identify discontinuities in phase data based on second derivatives.

    Args:
        phases (np.ndarray): Array of phase values
        x (np.ndarray): Corresponding x values for the phases

    Returns:
        tuple: A tuple containing:
            - discontinuities (np.ndarray): Indices of identified discontinuities (start and end).
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

        return discontinuities
