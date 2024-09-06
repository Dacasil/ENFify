from matplotlib import pyplot as plt
from scipy.io import wavfile

from enfify.config import DATA_DIR
from enfify.preprocessing import downsample_scipy, downsample_ffmpeg


def test_downsample_scipy():
    path = DATA_DIR / "external" / "ENF-WHU-Dataset" / "H1" / "001.wav"
    sample_rate, sig = wavfile.read(path)
    sig_ds, downsample_rate = downsample_scipy(sig, sample_rate, 1_000)
    plt.plot(sig_ds)
    plt.show()


def test_downsample_ffmpeg():
    path = DATA_DIR / "external" / "ENF-WHU-Dataset" / "H1" / "001.wav"
    sample_rate, sig = wavfile.read(path)
    sig_ds, downsample_rate = downsample_ffmpeg(sig, sample_rate, 1_000)
    plt.plot(sig_ds)
    plt.show()


if __name__ == "__main__":
    test_downsample_ffmpeg()
