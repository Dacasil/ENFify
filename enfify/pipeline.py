import numpy as np

from .preprocessing import butterworth_bandpass_filter, downsample_ffmpeg
from .feature_calculation import framing, freq_estimation_DFT1


def feature_freq_pipeline(sig, sample_freq, config):
    # Downsampling
    downsample_freq = config["downsample_per_enf"] * config["nominal_enf"]
    sig, sample_freq = downsample_ffmpeg(sig, sample_freq, downsample_freq)

    # Bandpass Filter
    lowcut = config["nominal_enf"] - config["bandpass_delta"]
    highcut = config["nominal_enf"] + config["bandpass_delta"]
    bandpass_order = config["bandpass_order"]
    sig = butterworth_bandpass_filter(sig, sample_freq, lowcut, highcut, bandpass_order)

    # Frame Splitting
    window_type = config["window_type"]
    frame_len = config["frame_len"]
    frame_shift = config["frame_step"]
    frames = framing(sig, sample_freq, frame_len, frame_shift, window_type)

    # Frequency Estimation
    n_dft = config["n_dft"]
    window_type = config["window_type"]
    feature_freq = np.array(
        [freq_estimation_DFT1(frame, sample_freq, n_dft, window_type) for frame in frames]
    )

    return feature_freq
