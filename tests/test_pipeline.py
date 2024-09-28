import unittest
import numpy as np
from unittest.mock import patch
from enfify.pipeline import feature_freq_pipeline


class TestFeatureFreqPipeline(unittest.TestCase):

    @patch("enfify.pipeline.downsample_ffmpeg")
    @patch("enfify.pipeline.butterworth_bandpass_filter")
    @patch("enfify.pipeline.framing")
    @patch("enfify.pipeline.freq_estimation_DFT1")
    def test_feature_freq_pipeline(
        self,
        mock_freq_estimation_DFT1,
        mock_framing,
        mock_butterworth_bandpass_filter,
        mock_downsample_ffmpeg,
    ):
        # Mocking the input signal and sample frequency
        sig = np.random.randn(1000)
        sample_freq = 1000

        # Mocking the configuration
        config = {
            "downsample_per_enf": 2,
            "nominal_enf": 50,
            "bandpass_delta": 1,
            "bandpass_order": 4,
            "window_type": "hamming",
            "frame_len": 256,
            "frame_step": 128,
            "n_dft": 512,
        }

        # Mocking the downsampled signal and frequency
        downsampled_sig = np.random.randn(500)
        downsampled_freq = 500
        mock_downsample_ffmpeg.return_value = (downsampled_sig, downsampled_freq)

        # Mocking the filtered signal
        filtered_sig = np.random.randn(500)
        mock_butterworth_bandpass_filter.return_value = filtered_sig

        # Mocking the frames
        frames = [np.random.randn(256) for _ in range(10)]
        mock_framing.return_value = frames

        # Mocking the frequency estimation
        mock_freq_estimation_DFT1.side_effect = lambda frame, sf, n_dft, wt: np.mean(frame)

        # Running the pipeline
        feature_freq = feature_freq_pipeline(sig, sample_freq, config)

        # Assertions
        mock_downsample_ffmpeg.assert_called_once_with(
            sig, sample_freq, config["downsample_per_enf"] * config["nominal_enf"]
        )
        mock_butterworth_bandpass_filter.assert_called_once_with(
            downsampled_sig,
            downsampled_freq,
            config["nominal_enf"] - config["bandpass_delta"],
            config["nominal_enf"] + config["bandpass_delta"],
            config["bandpass_order"],
        )
        mock_framing.assert_called_once_with(
            filtered_sig,
            downsampled_freq,
            config["frame_len"],
            config["frame_step"],
            config["window_type"],
        )
        self.assertEqual(len(feature_freq), len(frames))
        for frame, freq in zip(frames, feature_freq):
            self.assertEqual(freq, np.mean(frame))


if __name__ == "__main__":
    unittest.main()
