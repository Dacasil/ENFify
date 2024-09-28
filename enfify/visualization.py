from datetime import datetime
import matplotlib.pyplot as plt
from enfify.config import FIGURES_DIR


def report():
    raise NotImplementedError


def plot_feature_freq(feature_freq, filename=""):
    """Plot the frequency vector.

    Args:
        feature_freq (numpy.ndarray): Array of feature frequencies.
    """

    plt.plot(feature_freq)
    plt.xlabel("Feature index")
    plt.ylabel("Frequency")
    plt.title(f"Feature frequency - {filename}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(FIGURES_DIR / f"{filename}_{timestamp}.png", dpi=300)
