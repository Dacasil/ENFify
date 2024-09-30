import numpy as np

from enfify.feature_calculation import framing, freq_estimation_DFT1
from enfify.preprocessing import (
    butterworth_bandpass_filter,
    downsample_ffmpeg,
    downsample_scipy_new,
    fir_bandpass_filter,
)


# TODO: Remove when not needed
# def feature_freq_bilstm_pipeline(sig, sample_freq, config):
#     """
#     Processes an audio signal to extract frequency features for a CNN BiLSTM pipeline.

#     Args:
#         sig (numpy.ndarray): The input audio signal
#         sample_freq (float): The sampling frequency of the input signal
#         config (dict): Configuration dictionary containing parameters for processing

#     Returns:
#         tuple: A tuple containing:
#             - numpy.ndarray: The extracted spatial features.
#             - numpy.ndarray: The extracted temporal features.

#     Process:
#         - Downsamples the signal if enabled.
#         - Applies a bandpass filter if spenabled.
#         - Performs Variational Mode Decomposition (VMD) if enabled.
#         - Applies a Robust Filtering Algorithm (RFA) if enabled.
#         - Estimates the instantaneous frequencies using STFT.
#         - Trims the boundary values from the frequency estimates.
#         - Extracts spatial and temporal features for CNN BiLSTM processing.
#     """

#     feature_freq = feature_freq_pipeline(sig, sample_freq, config)

#     # Cut the boundary to weaken boundary value problems
#     feature_freq = feature_freq[40:-40]

#     # CNN BiLSTM feature processing
#     sn = config["bilstm_sn"]
#     fl = config["bilstm_fl"]
#     fn = config["bilstm_fn"]

#     spatial_features = extract_spatial_features(feature_freq, sn)
#     temporal_features = extract_temporal_features(feature_freq, fl, fn)

#     return spatial_features, temporal_features


def get_cycles(sig, sample_freq, config):
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

    return frames


def feature_freq_pipeline(sig, sample_freq, config):
    # """
    # Processes an audio signal to extract frequency features.

    # Args:
    #     sig (numpy.ndarray): The input audio signal
    #     sample_freq (float): The sampling frequency of the input signal
    #     config (dict): Configuration dictionary containing parameters for processing

    # Returns:
    #     - numpy.ndarray: The extracted spatial features.

    # Process:
    #     - Downsamples the signal.
    #     - Applies a bandpass filter.
    #     - Estimates the instantaneous frequencies.
    #     - Trims the boundary values from the frequency estimates.
    # """
    frames = get_cycles(sig, sample_freq, config)

    # Frequency Estimation
    n_dft = config["n_dft"]
    window_type = config["window_type"]
    feature_freq = np.array(
        [freq_estimation_DFT1(frame, sample_freq, n_dft, window_type) for frame in frames]
    )

    trim = config["feature_trim"]
    feature_freq = feature_freq[trim:-trim]

    return feature_freq


def feature_phase_pipeline(sig, sample_freq, config):
    frames = get_cycles(sig, sample_freq, config)

    # Phase Estimation
    raise NotImplementedError


# TODO: Remove when not needed
if __name__ == "__main__":
    from pathlib import Path

    import yaml
    from scipy.io import wavfile

    # from enfify.visualization import plot_feature_freq

    path = Path("/home/cloud/enfify/data/interim/Carioca1/HC01-00-tamp.wav")
    sample_freq, sig = wavfile.read(path)
    with open("/home/cloud/enfify/config/default.yml", "r") as f:
        config = yaml.safe_load(f)
    with open("/home/cloud/enfify/config/config_carioca.yml", "r") as f:
        config.update(yaml.safe_load(f))
    feature_freq = feature_freq_pipeline(sig, sample_freq, config)
    # plot_feature_freq(feature_freq, path.name)
