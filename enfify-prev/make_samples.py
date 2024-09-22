import io
from functools import reduce

import requests
from scipy.io import wavfile

from enfify import DATA_DIR, downsample_ffmpeg
from enfify.synthetic_signals import random_signal


def _make_samples():
    samples_dir = DATA_DIR / "samples"

    def make_whu_sample_files():
        def download_whu_01():
            url = "https://github.com/ghua-ac/ENF-WHU-Dataset/raw/78ed7f3784949f769f291fc1cb94acd10da6322f/ENF-WHU-Dataset/H1_ref/001_ref.wav"

            response = requests.get(url)

            if response.status_code != 200:
                raise Exception(f"Failed to download file. Status code: {response.status_code}")

            file_bytes = io.BytesIO(response.content)

            sample_freq, sig = wavfile.read(file_bytes)

            return sig, sample_freq

        os.makedirs(samples_dir, exist_ok=True)
        uncut_path = samples_dir / "whu_uncut_min_001_ref.wav"
        cut_path = samples_dir / "whu_cut_min_001_ref.wav"

        # downloading
        sig, sample_freq = download_whu_01()
        downsample_scipy(sig, sample_freq, 1_000)
        sig = sig[: sample_freq * 60]

        # Save uncut file
        wavfile.write(uncut_path, sample_freq, sig)

        # cutting
        location = np.random.randint(0, len(sig) - sample_freq * 10)
        cutlen = np.random.randint(10 * sample_freq)
        sig = np.delete(sig, slice(location, location + cutlen))

        # Save cut file
        wavfile.write(cut_path, sample_freq, sig)

    def make_synthetic_sample_files():
        os.makedirs(samples_dir, exist_ok=True)
        uncut_path = samples_dir / "synthetic_uncut_0.wav"
        cut_path = samples_dir / "synthetic_cut_0.wav"

        sig_uncut, sig_cut = random_signal(1, 60, 1_000, 50, 0.2, 10_000)

        wavfile.write(uncut_path, 1_000, sig_uncut)
        wavfile.write(cut_path, 1_000, sig_cut)

    make_whu_sample_files()

    make_synthetic_sample_files()
